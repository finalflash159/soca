from pathlib import Path

import pytest

from soca.knowledge import MarkdownVaultKnowledgeSource


def test_reads_markdown_file(tmp_path: Path):
    (tmp_path / "note.md").write_text(
        "# PhoGPT\nNội dung về LLM bakeoff.\n\n#llm #bakeoff",
        encoding="utf-8",
    )

    vault = MarkdownVaultKnowledgeSource(tmp_path)
    doc = vault.read("note.md")

    assert doc.path == "note.md"
    assert doc.title == "PhoGPT"
    assert doc.tags == ("bakeoff", "llm")
    assert "LLM bakeoff" in doc.text


def test_rejects_path_traversal(tmp_path: Path):
    vault = MarkdownVaultKnowledgeSource(tmp_path)

    with pytest.raises(ValueError):
        vault.read("../secret.md")


def test_excludes_private_directory(tmp_path: Path):
    private = tmp_path / "private"
    private.mkdir()
    (private / "secret.md").write_text("token", encoding="utf-8")

    vault = MarkdownVaultKnowledgeSource(tmp_path)

    with pytest.raises(ValueError):
        vault.read("private/secret.md")


def test_excludes_dot_directories_from_read_and_search(tmp_path: Path):
    obsidian = tmp_path / ".obsidian"
    obsidian.mkdir()
    (obsidian / "config.md").write_text("# Config\nObsidian internals", encoding="utf-8")

    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "secret.md").write_text("# Secret\nRepository internals", encoding="utf-8")

    (tmp_path / "public.md").write_text("# Public\nObsidian runtime note.", encoding="utf-8")

    vault = MarkdownVaultKnowledgeSource(tmp_path)

    with pytest.raises(ValueError):
        vault.read(".obsidian/config.md")

    with pytest.raises(ValueError):
        vault.read(".git/secret.md")

    hits = vault.search("internals Obsidian")

    assert [hit.document.path for hit in hits] == ["public.md"]


def test_search_vietnamese_terms_returns_snippet(tmp_path: Path):
    (tmp_path / "phase3.md").write_text(
        "# Phase 3D\nThiết kế Obsidian knowledge cho SoCa.",
        encoding="utf-8",
    )

    vault = MarkdownVaultKnowledgeSource(tmp_path)
    hits = vault.search("Obsidian knowledge")

    assert len(hits) == 1
    assert hits[0].document.path == "phase3.md"
    assert "Obsidian" in hits[0].snippet


def test_search_matches_vietnamese_without_accents(tmp_path: Path):
    (tmp_path / "breakfast.md").write_text(
        "# Bữa Sáng Lành Mạnh\nYến mạch và sữa chua giúp bữa sáng cân bằng.",
        encoding="utf-8",
    )

    vault = MarkdownVaultKnowledgeSource(tmp_path)
    hits = vault.search("bua sang lanh manh")

    assert [hit.document.path for hit in hits] == ["breakfast.md"]


def test_snippet_prefers_body_over_heading_and_tags(tmp_path: Path):
    (tmp_path / "meal.md").write_text(
        "# Bữa Sáng Lành Mạnh\n\n"
        "#dinh-duong #bua-sang\n\n"
        "Bạn có thể ăn yến mạch với sữa chua không đường và chuối.",
        encoding="utf-8",
    )

    vault = MarkdownVaultKnowledgeSource(tmp_path)
    hits = vault.search("bữa sáng lành mạnh")

    assert len(hits) == 1
    assert "yến mạch" in hits[0].snippet
    assert "#dinh-duong" not in hits[0].snippet


def test_snippet_prefers_strongest_body_overlap(tmp_path: Path):
    (tmp_path / "meal.md").write_text(
        "# Gợi Ý Bữa Ăn Lành Mạnh\n\n"
        "Các gợi ý dưới đây dùng cho người trưởng thành khỏe mạnh muốn ăn cân bằng hơn.\n\n"
        'Người dùng: "Tôi muốn bữa sáng nhanh mà lành mạnh."\n\n'
        "Assistant: Bạn có thể ăn yến mạch với sữa chua không đường và chuối.",
        encoding="utf-8",
    )

    vault = MarkdownVaultKnowledgeSource(tmp_path)
    hits = vault.search("tôi muốn bữa sáng nhanh lành mạnh")

    assert len(hits) == 1
    assert "yến mạch" in hits[0].snippet


def test_include_globs_can_scope_runtime_to_compiled_wiki(tmp_path: Path):
    raw = tmp_path / "raw" / "sources"
    raw.mkdir(parents=True)
    (raw / "source.md").write_text("# Raw\nPhoGPT raw source.", encoding="utf-8")

    wiki = tmp_path / "wiki" / "decisions"
    wiki.mkdir(parents=True)
    (wiki / "llm-bakeoff.md").write_text("# LLM Bakeoff\nPhoGPT compiled wiki.", encoding="utf-8")

    vault = MarkdownVaultKnowledgeSource(tmp_path, include_globs=("wiki/**/*.md",))
    hits = vault.search("PhoGPT")

    assert [hit.document.path for hit in hits] == ["wiki/decisions/llm-bakeoff.md"]


def test_search_skips_default_index_and_log_files(tmp_path: Path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "index.md").write_text("# Index\nPhoGPT runtime", encoding="utf-8")
    (wiki / "log.md").write_text("# Log\nPhoGPT runtime", encoding="utf-8")
    (wiki / "runtime.md").write_text("# Runtime\nPhoGPT runtime", encoding="utf-8")

    vault = MarkdownVaultKnowledgeSource(tmp_path, include_globs=("wiki/**/*.md",))
    hits = vault.search("PhoGPT runtime")

    assert [hit.document.path for hit in hits] == ["wiki/runtime.md"]
    assert vault.read("wiki/index.md").title == "Index"


def test_search_results_are_sorted_by_score_then_path(tmp_path: Path):
    (tmp_path / "b.md").write_text("# Runtime\nPhoGPT", encoding="utf-8")
    (tmp_path / "a.md").write_text("# Runtime\nPhoGPT", encoding="utf-8")
    (tmp_path / "strong.md").write_text("# PhoGPT Runtime\nPhoGPT runtime", encoding="utf-8")

    vault = MarkdownVaultKnowledgeSource(tmp_path)
    hits = vault.search("PhoGPT runtime")

    assert [hit.document.path for hit in hits] == ["strong.md", "a.md", "b.md"]


def test_search_prefers_specific_metadata_without_query_stuffing(tmp_path: Path):
    (tmp_path / "todo.md").write_text(
        "# Todo Hôm Nay\n\n"
        "#todo #checklist #tap-luyen #dinh-duong\n\n"
        "Mỗi ngày nên có việc chính, sức khỏe, và check-in ngắn.",
        encoding="utf-8",
    )
    (tmp_path / "nutrition.md").write_text(
        "# Dinh Dưỡng Cho Người Tập Luyện\n\n"
        "#nutrition #protein #tap-luyen #dinh-duong\n\n"
        "Người tập luyện nên ưu tiên đủ đạm, rau, tinh bột phù hợp và nước.",
        encoding="utf-8",
    )

    vault = MarkdownVaultKnowledgeSource(tmp_path)
    hits = vault.search("todo hôm nay tập luyện dinh dưỡng")

    assert [hit.document.path for hit in hits[:2]] == ["todo.md", "nutrition.md"]


def test_search_skips_large_file(tmp_path: Path):
    (tmp_path / "large.md").write_text("PhoGPT\n" * 1000, encoding="utf-8")

    vault = MarkdownVaultKnowledgeSource(tmp_path, max_file_bytes=10)

    assert vault.search("PhoGPT") == []
