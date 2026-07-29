from __future__ import annotations

import pytest

from eval.retrieval_bakeoff import (
    QueryMeasurement,
    ndcg_at_k,
    select_query_ids,
    summarize_measurements,
)


def test_ndcg_supports_graded_relevance() -> None:
    qrels = {"best": 3, "useful": 1}

    ideal = ndcg_at_k(("best", "useful"), qrels, k=10)
    reversed_score = ndcg_at_k(("useful", "best"), qrels, k=10)

    assert ideal == pytest.approx(1.0)
    assert reversed_score < ideal


def test_query_selection_is_deterministic_and_only_uses_judged_queries() -> None:
    qrels = {f"q{index}": {"doc": 1} for index in range(20)}

    first = select_query_ids(qrels, limit=5, seed=42)
    second = select_query_ids(qrels, limit=5, seed=42)

    assert first == second
    assert len(first) == 5
    assert set(first) <= set(qrels)


def test_summary_reports_quality_and_latency_percentiles() -> None:
    samples = (
        QueryMeasurement("q1", 1.0, 1.0, 1.0, 1.0, 2.0),
        QueryMeasurement("q2", 0.0, 0.0, 0.0, 0.0, 10.0),
    )

    summary = summarize_measurements(samples)

    assert summary["recall_at_5"] == pytest.approx(0.5)
    assert summary["mrr_at_10"] == pytest.approx(0.5)
    assert summary["latency_p50_ms"] == pytest.approx(2.0)
    assert summary["latency_p95_ms"] == pytest.approx(10.0)
