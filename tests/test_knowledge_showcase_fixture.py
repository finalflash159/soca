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
        "wiki/life/finance/food-budget-2026-07.md",
        "wiki/life/finance/receipt-ledger-2026-07.md",
        "wiki/life/health/safety-boundaries.md",
        "wiki/life/health/questions-for-a-clinician.md",
        "wiki/life/journal/2026-07-28.md",
        "wiki/life/journal/2026-07-29.md",
        "wiki/learning/data/data-pipelines.md",
        "wiki/learning/networks/http-timeouts-and-retries.md",
        "wiki/learning/security/threat-modeling.md",
        "wiki/xquad_vi/Oxygen.md",
        "wiki/xquad_vi/Computational_complexity_theory.md",
    }.issubset(paths)
    assert not any(path.startswith("wiki/dinh-duong/") for path in paths)
    assert not any(path.startswith("wiki/life/project/") for path in paths)
    assert not any(path.startswith("wiki/life/decisions/") for path in paths)
    assert "wiki/learning/llm/context-and-tool-use.md" not in paths
    learning_paths = [
        path
        for path in paths
        if path.startswith("wiki/learning/") and not path.startswith("wiki/xquad_vi/")
    ]
    public_paths = [path for path in paths if path.startswith("wiki/xquad_vi/")]
    life_paths = [path for path in paths if path.startswith("wiki/life/")]
    assert len(learning_paths) >= 15
    assert len(public_paths) == 48
    assert len(life_paths) >= 15
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
    dalat_split = source.search(
        "chuyến Đà Lạt bữa trước chia tiền tới đâu còn ai chưa chuyển",
        limit=3,
    )
    attention_follow_up = source.search(
        "attention với transformer hôm trước mình đang hiểu tới đâu rồi",
        limit=3,
    )
    serving_follow_up = source.search(
        "lúc học về serving local với remote mình rút ra nguyên tắc gì",
        limit=3,
    )

    assert bayes[0].document.path == "wiki/learning/fundamentals/bayes-and-probability.md"
    assert budget[0].document.path == "wiki/life/finance/food-budget-2026-07.md"
    assert "1.843.000" in budget[0].document.text
    assert "2.209.000" in budget[0].document.text
    assert dalat_split[0].document.path == "wiki/life/finance/da-lat-trip-2026-07.md"
    assert attention_follow_up[0].document.path == (
        "wiki/learning/deep-learning/attention-and-transformers.md"
    )
    assert serving_follow_up[0].document.path == (
        "wiki/learning/llm/serving-local-remote.md"
    )
    assert source.search("Sao Bắc Cực X9", limit=3) == []
