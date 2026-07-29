from __future__ import annotations

from pathlib import Path

from soca.knowledge import MarkdownVaultKnowledgeSource

ROOT = Path(__file__).resolve().parents[1]
SHOWCASE_VAULT = ROOT / "eval" / "fixtures" / "knowledge_vault"


def test_showcase_vault_has_realistic_demo_slices() -> None:
    source = MarkdownVaultKnowledgeSource(
        SHOWCASE_VAULT,
        include_globs=("wiki/**/*.md",),
    )

    paths = source.iter_paths()

    assert len(paths) >= 16
    assert {
        "wiki/learning/notes/bayes-theorem.md",
        "wiki/life/project/tts-decision.md",
        "wiki/life/finance/food-budget-2026-07.md",
        "wiki/life/health/safety-boundaries.md",
    }.issubset(paths)
    assert not any(path.startswith("wiki/dinh-duong/") for path in paths)
    assert all(path.startswith("wiki/") for path in paths)


def test_showcase_vault_has_answerable_and_unanswerable_smoke_queries() -> None:
    source = MarkdownVaultKnowledgeSource(
        SHOWCASE_VAULT,
        include_globs=("wiki/**/*.md",),
    )

    bayes = source.search("định lý Bayes", limit=3)
    budget = source.search("ngân sách ăn uống tháng 07/2026", limit=3)

    assert bayes[0].document.path == "wiki/learning/notes/bayes-theorem.md"
    assert budget[0].document.path == "wiki/life/finance/food-budget-2026-07.md"
    assert source.search("Sao Bắc Cực X9", limit=3) == []
