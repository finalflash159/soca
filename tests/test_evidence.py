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
