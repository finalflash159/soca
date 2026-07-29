from __future__ import annotations

import pytest

from eval.retrieval_bakeoff import (
    LinearHybridRanker,
    QueryMeasurement,
    StaticRanker,
    ndcg_at_k,
    parse_candidate,
    select_query_ids,
    summarize_measurements,
)
from soca.knowledge.retriever import RankedHit


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


def test_parse_candidate_supports_explicit_fusion_and_rerank() -> None:
    rrf = parse_candidate("hybrid_rrf:bm25:fastembed")
    linear = parse_candidate("hybrid_linear:bm25:fastembed:0.35")
    reranked = parse_candidate(
        "rerank:hybrid_rrf:bm25:fastembed:rerank_mmarco_minilm:20"
    )

    assert rrf.fusion == "rrf"
    assert linear.fusion == "linear"
    assert linear.dense_weight == pytest.approx(0.35)
    assert reranked.reranker == "rerank_mmarco_minilm"
    assert reranked.rerank_top_k == 20


def test_linear_fusion_normalizes_backend_scores_before_weighting() -> None:
    sparse = StaticRanker(
        (
            RankedHit("sparse-only", 1, 1000.0),
            RankedHit("both", 2, 100.0),
        )
    )
    dense = StaticRanker(
        (
            RankedHit("dense-only", 1, 0.9),
            RankedHit("both", 2, 0.8),
        )
    )

    hits = LinearHybridRanker(sparse, dense, dense_weight=0.75).rank(
        "query",
        limit=3,
    )

    assert [hit.chunk_id for hit in hits] == ["dense-only", "sparse-only", "both"]
    assert hits[0].score == pytest.approx(0.75)
    assert hits[1].score == pytest.approx(0.25)
