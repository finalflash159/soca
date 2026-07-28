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
