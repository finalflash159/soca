"""Calibratable retrieval-evidence contracts, distinct from route confidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EvidenceStatus = Literal["supported", "insufficient", "unavailable"]


@dataclass(frozen=True)
class EvidenceDecision:
    source: Literal["knowledge", "memory"]
    status: EvidenceStatus
    hit_count: int
    top_score: float | None
    reason: str


@dataclass(frozen=True)
class EvidenceBundleDecision:
    status: Literal["consistent", "conflicting", "unknown"]
    decisions: tuple[EvidenceDecision, ...]
    reason: str


class EvidenceReconciler:
    """Reconcile source-local decisions without comparing raw corpus scores."""

    def reconcile(self, decisions: tuple[EvidenceDecision, ...]) -> EvidenceBundleDecision:
        supported = tuple(item for item in decisions if item.status == "supported")
        if not supported:
            return EvidenceBundleDecision("unknown", decisions, "no_supported_source")
        if len(supported) == 1:
            return EvidenceBundleDecision("consistent", decisions, "one_supported_source")
        # Textual contradiction detection belongs to a frozen later evaluator;
        # source score magnitude is not meaningful across independent corpora.
        return EvidenceBundleDecision(
            "unknown", decisions, "multiple_supported_sources_unreconciled"
        )


def decide_evidence(
    source: Literal["knowledge", "memory"],
    hits: tuple[object, ...],
    *,
    unavailable: bool = False,
) -> EvidenceDecision:
    if unavailable:
        return EvidenceDecision(source, "unavailable", 0, None, "retrieval_unavailable")
    if not hits:
        return EvidenceDecision(source, "insufficient", 0, None, "no_hits")
    score = getattr(hits[0], "score", None)
    raw_score = getattr(score, "total", score)
    if raw_score is None:
        top_score = None
    else:
        try:
            top_score = float(raw_score)
        except (TypeError, ValueError):
            top_score = None
    return EvidenceDecision(source, "supported", len(hits), top_score, "hits_present")


__all__ = [
    "EvidenceBundleDecision",
    "EvidenceDecision",
    "EvidenceReconciler",
    "EvidenceStatus",
    "decide_evidence",
]
