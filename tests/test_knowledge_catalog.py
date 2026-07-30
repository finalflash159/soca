from __future__ import annotations

import json
from pathlib import Path

from soca.knowledge.catalog import (
    CatalogIndexSnapshot,
    KnowledgeCatalog,
    build_catalog_snapshot,
)
from soca.knowledge.indexing.scanner import scan_vault
from soca.knowledge.markdown_vault import MarkdownVaultKnowledgeSource
from soca.tools import KnowledgeInspectTool, ToolCall, ToolRuntime


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _index(root: Path):
    source = MarkdownVaultKnowledgeSource(
        root,
        include_globs=("wiki/**/*.md",),
        exclude_files=("log.md",),
    )
    return source, scan_vault(source).index


def test_catalog_builds_folder_heading_link_and_backlink_graph(tmp_path: Path) -> None:
    _write(
        tmp_path / "wiki/learning/attention.md",
        "# Attention\n\n## Query và key\nNội dung.\n\n"
        "Xem [[transformer|Transformer]] và [Bayes](../math/bayes.md#formula).\n",
    )
    _write(
        tmp_path / "wiki/learning/transformer.md",
        "# Transformer\n\n## Blocks\nNội dung.\n",
    )
    _write(
        tmp_path / "wiki/math/bayes.md",
        "# Bayes\n\n## Formula\nNội dung.\n\n[External](https://example.com).\n",
    )
    _, index = _index(tmp_path)

    snapshot = build_catalog_snapshot(index, revision=7)

    assert snapshot.revision == 7
    assert snapshot.folders == ("wiki", "wiki/learning", "wiki/math")
    attention = next(item for item in snapshot.documents if item.path.endswith("attention.md"))
    assert [(heading.level, heading.text) for heading in attention.headings] == [
        (1, "Attention"),
        (2, "Query và key"),
    ]
    assert {
        (relation.source, relation.target, relation.kind)
        for relation in snapshot.relations
    } == {
        (
            "wiki/learning/attention.md",
            "wiki/learning/transformer.md",
            "wikilink",
        ),
        (
            "wiki/learning/attention.md",
            "wiki/math/bayes.md",
            "markdown_link",
        ),
    }
    neighborhood = snapshot.neighborhood(("wiki/learning/transformer.md",))
    assert {item.path for item in neighborhood.documents} == {
        "wiki/learning/attention.md",
        "wiki/learning/transformer.md",
    }
    assert neighborhood.relations[0].source == "wiki/learning/attention.md"
    assert snapshot.unresolved_links == ()


def test_manifest_is_navigation_metadata_without_retrieval_hit_shape(tmp_path: Path) -> None:
    _write(tmp_path / "wiki/learning/attention.md", "# Attention\n\nNote.\n")
    _, index = _index(tmp_path)

    manifest = build_catalog_snapshot(index, revision=8).manifest_dict()

    assert manifest["revision"] == 8
    assert manifest["document_count"] == 1
    assert manifest["tree"] == {"wiki/learning": ["wiki/learning/attention.md"]}
    assert "hits" not in manifest
    assert "evidence_status" not in manifest


def test_catalog_reports_unresolved_internal_links_without_inventing_edges(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "wiki/a.md",
        "# A\n\n[[missing-note]]\n\n![image](asset.png)\n",
    )
    _, index = _index(tmp_path)

    snapshot = build_catalog_snapshot(index, revision=1)

    assert snapshot.relations == ()
    assert [(item.source, item.target, item.kind) for item in snapshot.unresolved_links] == [
        ("wiki/a.md", "missing-note", "wikilink")
    ]


def test_catalog_ignores_link_syntax_inside_code(tmp_path: Path) -> None:
    _write(
        tmp_path / "wiki/a.md",
        "# A\n\n`[[inline-code]]`\n\n"
        "```md\n[[fenced-code]]\n[also code](missing.md)\n```\n\n"
        "[Real](b.md)\n",
    )
    _write(tmp_path / "wiki/b.md", "# B\n")
    _, index = _index(tmp_path)

    snapshot = build_catalog_snapshot(index, revision=1)

    assert [
        (item.source, item.target, item.kind)
        for item in snapshot.relations
    ] == [("wiki/a.md", "wiki/b.md", "markdown_link")]
    assert snapshot.unresolved_links == ()


def test_catalog_prompt_budget_keeps_inventory_and_reports_truncation(
    tmp_path: Path,
) -> None:
    for index in range(8):
        headings = "\n".join(
            f"## Heading {heading} {'chi tiết ' * 10}"
            for heading in range(12)
        )
        _write(
            tmp_path / f"wiki/group/note-{index}.md",
            f"# Note {index}\n\n{headings}\n",
        )
    _, index = _index(tmp_path)
    snapshot = build_catalog_snapshot(index, revision=2)

    text = snapshot.prompt_text(max_chars=4_096)
    payload = json.loads(text.splitlines()[-1])

    assert len(text) <= 4_096
    assert payload["truncated"] is True
    assert len(payload["documents"]) == 8
    assert {item["path"] for item in payload["documents"]} == {
        f"wiki/group/note-{index}.md" for index in range(8)
    }


def test_inspect_expands_explicit_relations_without_creating_evidence(tmp_path: Path) -> None:
    _write(tmp_path / "wiki/a.md", "# A\n\nXem [[b]].\n")
    _write(tmp_path / "wiki/b.md", "# B\n\nXem [[c]].\n")
    _write(tmp_path / "wiki/c.md", "# C\n")
    source, index = _index(tmp_path)

    class Provider:
        revision = 4

        def catalog_index_snapshot(self) -> CatalogIndexSnapshot:
            return CatalogIndexSnapshot(self.revision, index)

    result = ToolRuntime([KnowledgeInspectTool(KnowledgeCatalog(Provider()))]).call(
        ToolCall("knowledge.inspect", {"path": "wiki/a.md", "depth": 1})
    )

    assert result.ok
    assert result.data["metadata_only"] is True
    assert result.data["depth"] == 1
    assert [item["path"] for item in result.data["documents"]] == [
        "wiki/a.md",
        "wiki/b.md",
    ]
    assert result.data["relations"] == [
        {"source": "wiki/a.md", "target": "wiki/b.md", "kind": "wikilink"}
    ]
    assert "hits" not in result.data


def test_inspect_degrades_metadata_shape_before_failing_context_budget(tmp_path: Path) -> None:
    for index in range(20):
        _write(
            tmp_path / f"wiki/notes/note-{index}.md",
            f"# Note {index}\n\n" + "\n".join(
                f"## Heading {heading} {'chi tiết ' * 12}" for heading in range(8)
            ),
        )
    _, index = _index(tmp_path)

    class Provider:
        def catalog_index_snapshot(self) -> CatalogIndexSnapshot:
            return CatalogIndexSnapshot(5, index)

    result = ToolRuntime(
        [KnowledgeInspectTool(KnowledgeCatalog(Provider()), max_chars=2_048)]
    ).call(ToolCall("knowledge.inspect", {}))

    assert result.ok
    assert result.data["truncated"] is True
    assert all(item.get("headings_omitted") for item in result.data["documents"])
