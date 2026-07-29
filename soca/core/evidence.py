"""Calibratable retrieval-evidence contracts, distinct from route confidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EvidenceStatus = Literal[
    "supported",
    "weak",
    "conflicting",
    "insufficient",
    "unavailable",
]
EvidenceSourceState = Literal[
    "ready",
    "empty",
    "degraded",
    "missing",
    "stale",
    "unavailable",
    "unknown",
]
EvidenceRelation = Literal["consistent", "conflicting", "unknown"]


@dataclass(frozen=True)
class EvidenceDecision:
    source: Literal["knowledge", "memory"]
    status: EvidenceStatus
    hit_count: int
    top_score: float | None
    margin: float | None
    rejected_count: int
    reason: str
    source_state: EvidenceSourceState = "unknown"
    query_coverage: float | None = None
    score_separation: float | None = None
    sparse_top_score: float | None = None
    dense_top_score: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "status": self.status,
            "hit_count": self.hit_count,
            "top_score": self.top_score,
            "margin": self.margin,
            "rejected_count": self.rejected_count,
            "reason": self.reason,
            "source_state": self.source_state,
            "query_coverage": self.query_coverage,
            "score_separation": self.score_separation,
            "sparse_top_score": self.sparse_top_score,
            "dense_top_score": self.dense_top_score,
        }


@dataclass(frozen=True)
class EvidenceBundleDecision:
    status: Literal["consistent", "conflicting", "unknown"]
    decisions: tuple[EvidenceDecision, ...]
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "decisions": [decision.as_dict() for decision in self.decisions],
            "reason": self.reason,
        }


class EvidenceReconciler:
    """Reconcile source-local decisions without comparing raw corpus scores."""

    def reconcile(
        self,
        decisions: tuple[EvidenceDecision, ...],
        *,
        relation: EvidenceRelation = "unknown",
    ) -> EvidenceBundleDecision:
        if relation == "conflicting":
            return EvidenceBundleDecision("conflicting", decisions, "evaluator_found_conflict")
        if relation == "consistent":
            return EvidenceBundleDecision("consistent", decisions, "evaluator_found_consistency")
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
    status: EvidenceStatus | None = None,
    reason: str | None = None,
    top_score: float | None = None,
    margin: float | None = None,
    rejected_count: int = 0,
    source_state: EvidenceSourceState = "unknown",
    query_coverage: float | None = None,
    score_separation: float | None = None,
    sparse_top_score: float | None = None,
    dense_top_score: float | None = None,
) -> EvidenceDecision:
    if unavailable:
        return EvidenceDecision(
            source,
            "unavailable",
            0,
            None,
            None,
            rejected_count,
            "retrieval_unavailable",
            "unavailable",
            query_coverage,
            score_separation,
            sparse_top_score,
            dense_top_score,
        )
    if not hits:
        return EvidenceDecision(
            source,
            status or "insufficient",
            0,
            top_score,
            margin,
            rejected_count,
            reason or "no_hits",
            source_state,
            query_coverage,
            score_separation,
            sparse_top_score,
            dense_top_score,
        )
    if top_score is None:
        score = getattr(hits[0], "score", None)
        raw_score = getattr(score, "total", score)
        try:
            top_score = float(raw_score) if raw_score is not None else None
        except (TypeError, ValueError):
            top_score = None
    return EvidenceDecision(
        source,
        status or "supported",
        len(hits),
        top_score,
        margin,
        rejected_count,
        reason or "hits_present",
        source_state,
        query_coverage,
        score_separation,
        sparse_top_score,
        dense_top_score,
    )


__all__ = [
    "EvidenceBundleDecision",
    "EvidenceDecision",
    "EvidenceReconciler",
    "EvidenceStatus",
    "EvidenceSourceState",
    "EvidenceRelation",
    "decide_evidence",
]
