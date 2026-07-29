from __future__ import annotations

from dataclasses import dataclass

from soca.core.evidence import EvidenceReconciler, decide_evidence


@dataclass(frozen=True)
class _Hit:
    score: float


def test_evidence_decision_does_not_compare_scores_across_sources() -> None:
    knowledge = decide_evidence("knowledge", (_Hit(0.2),))
    memory = decide_evidence("memory", (_Hit(99.0),))
    decision = EvidenceReconciler().reconcile((knowledge, memory))
    assert decision.status == "unknown"
    assert decision.reason == "multiple_supported_sources_unreconciled"


def test_evidence_distinguishes_insufficient_from_unavailable() -> None:
    assert decide_evidence("knowledge", ()).status == "insufficient"
    assert decide_evidence("knowledge", (), unavailable=True).status == "unavailable"


def test_evidence_can_preserve_relevance_assessment_metadata() -> None:
    decision = decide_evidence(
        "knowledge",
        (_Hit(0.9),),
        status="weak",
        reason="ambiguous_top_margin",
        top_score=0.9,
        margin=0.01,
        rejected_count=2,
    )

    assert decision.status == "weak"
    assert decision.reason == "ambiguous_top_margin"
    assert decision.margin == 0.01
    assert decision.rejected_count == 2


def test_evidence_exposes_backend_local_signals_without_cross_source_comparison() -> None:
    decision = decide_evidence(
        "knowledge",
        (_Hit(0.9),),
        source_state="ready",
        query_coverage=0.5,
        score_separation=0.2,
        sparse_top_score=14.0,
        dense_top_score=0.81,
    )

    assert decision.query_coverage == 0.5
    assert decision.score_separation == 0.2
    assert decision.sparse_top_score == 14.0
    assert decision.dense_top_score == 0.81
    assert decision.as_dict()["source_state"] == "ready"


def test_evidence_reconciler_accepts_external_consistency_relation() -> None:
    decision = decide_evidence("knowledge", (_Hit(0.9),))

    consistent = EvidenceReconciler().reconcile((decision,), relation="consistent")
    conflicting = EvidenceReconciler().reconcile((decision,), relation="conflicting")

    assert consistent.status == "consistent"
    assert conflicting.status == "conflicting"
