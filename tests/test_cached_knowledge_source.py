from __future__ import annotations

from pathlib import Path

import pytest

from soca.knowledge.cached_source import CachedMarkdownVaultKnowledgeSource
from soca.knowledge.markdown_vault import (
    MarkdownVaultKnowledgeSource,
    SearchScoringConfig,
)


def _make_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    wiki = vault / "wiki"
    wiki.mkdir(parents=True)
    (wiki / "a.md").write_text(
        "# Runtime Notes\n"
        "#assistant #voice\n\n"
        "PhoGPT runtime supports Vietnamese voice conversations.",
        encoding="utf-8",
    )
    (wiki / "b.md").write_text(
        "# PhoGPT Runtime\n#llm\n\nThe runtime routes Vietnamese prompts to PhoGPT.",
        encoding="utf-8",
    )
    (wiki / "c.md").write_text(
        "# Nutrition\n#health\n\nYến mạch phù hợp cho bữa sáng lành mạnh.",
        encoding="utf-8",
    )
    return vault


@pytest.mark.parametrize(
    ("query", "scoring"),
    [
        ("PhoGPT runtime", SearchScoringConfig()),
        (
            "Vietnamese voice",
            SearchScoringConfig(title_weight=9.0, body_phrase_weight=7.0),
        ),
        ("bua sang lanh manh", SearchScoringConfig()),
        ("missing phrase", SearchScoringConfig()),
    ],
)
def test_cached_search_has_exact_path_score_snippet_and_order_parity(
    tmp_path: Path,
    query: str,
    scoring: SearchScoringConfig,
) -> None:
    vault = _make_vault(tmp_path)
    source_options = {
        "include_globs": ("wiki/**/*.md",),
        "scoring": scoring,
    }
    baseline = MarkdownVaultKnowledgeSource(vault, **source_options)
    cached = CachedMarkdownVaultKnowledgeSource(
        vault,
        index_home=tmp_path / "index-home",
        **source_options,
    )

    expected = [
        (hit.document.path, hit.score, hit.snippet) for hit in baseline.search(query, limit=5)
    ]
    actual = [(hit.document.path, hit.score, hit.snippet) for hit in cached.search(query, limit=5)]

    assert actual == expected


def test_unchanged_cached_search_does_not_read_markdown_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached = CachedMarkdownVaultKnowledgeSource(
        _make_vault(tmp_path),
        index_home=tmp_path / "index-home",
        include_globs=("wiki/**/*.md",),
    )
    first = cached.search("PhoGPT runtime")

    def fail_read(path: str) -> None:
        pytest.fail(f"unexpected Markdown read: {path}")

    monkeypatch.setattr(cached, "read", fail_read)

    assert cached.search("PhoGPT runtime") == first
