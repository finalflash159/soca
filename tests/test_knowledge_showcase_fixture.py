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

    assert len(paths) >= 35
    assert {
        "wiki/learning/fundamentals/bayes-and-probability.md",
        "wiki/learning/dsa/graphs-and-dp.md",
        "wiki/learning/systems/databases-and-indexes.md",
        "wiki/life/decisions/tts-choice.md",
        "wiki/life/decisions/knowledge-vault-shape.md",
        "wiki/life/finance/food-budget-2026-07.md",
        "wiki/life/finance/receipt-ledger-2026-07.md",
        "wiki/life/health/safety-boundaries.md",
        "wiki/life/health/questions-for-a-clinician.md",
        "wiki/life/health/sleep-observations.md",
        "wiki/life/journal/2026-07-28.md",
        "wiki/life/journal/2026-07-29.md",
        "wiki/learning/data/data-pipelines.md",
        "wiki/learning/networks/http-timeouts-and-retries.md",
        "wiki/learning/security/threat-modeling.md",
    }.issubset(paths)
    assert not any(path.startswith("wiki/dinh-duong/") for path in paths)
    assert not any(path.startswith("wiki/life/project/") for path in paths)
    learning_paths = [path for path in paths if path.startswith("wiki/learning/")]
    life_paths = [path for path in paths if path.startswith("wiki/life/")]
    assert len(learning_paths) >= 16
    assert len(life_paths) >= 18
    assert all(
        len((SHOWCASE_VAULT / path).read_text(encoding="utf-8").splitlines()) >= 100
        for path in learning_paths
    )
    assert all(
        len((SHOWCASE_VAULT / path).read_text(encoding="utf-8").splitlines()) >= 50
        for path in life_paths
    )
    assert all(path.startswith("wiki/") for path in paths)


def test_showcase_vault_has_answerable_and_unanswerable_smoke_queries() -> None:
    source = MarkdownVaultKnowledgeSource(
        SHOWCASE_VAULT,
        include_globs=("wiki/**/*.md",),
    )

    bayes = source.search("định lý Bayes", limit=3)
    budget = source.search("ngân sách ăn uống tháng 07/2026", limit=3)

    assert bayes[0].document.path == "wiki/learning/fundamentals/bayes-and-probability.md"
    assert budget[0].document.path == "wiki/life/finance/food-budget-2026-07.md"
    assert source.search("Sao Bắc Cực X9", limit=3) == []
