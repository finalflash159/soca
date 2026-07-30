from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from soca.knowledge.base import KnowledgeHit
from soca.knowledge.markdown_vault import tokenize_terms

RelevanceStatus = Literal["supported", "weak", "insufficient"]
_SignalSpace = Literal["explicit", "lexical", "sparse", "dense"]


@dataclass(frozen=True)
class RelevancePolicy:
    """Calibratable admission policy for retrieved evidence.

    Scores are backend-local signals. A dense cosine threshold is never
    compared with a lexical or fusion score; each backend contributes its own
    signal and the context builder records the resulting decision.
    """

    # Calibrated on the held-out public XQuAD grounding set: 0.65 kept
    # answerable recall while removing generic account/weather distractors.
    min_lexical_coverage: float = 0.65
    min_sparse_score_ratio: float = 0.75
    min_dense_score: float = 0.55
    min_top_margin: float = 0.05
    min_fusion_score_ratio: float = 0.70

    @classmethod
    def for_retrieval_mode(cls, mode: str) -> RelevancePolicy:
        """Return the runtime contract for a backend score distribution.

        Corpus/model calibration belongs in a versioned artifact. This default
        deliberately does not encode a threshold learned from another corpus.
        """
        del mode
        return cls(min_lexical_coverage=0.65, min_dense_score=0.55)

    def __post_init__(self) -> None:
        for name, value in (
            ("min_lexical_coverage", self.min_lexical_coverage),
            ("min_sparse_score_ratio", self.min_sparse_score_ratio),
            ("min_dense_score", self.min_dense_score),
            ("min_top_margin", self.min_top_margin),
            ("min_fusion_score_ratio", self.min_fusion_score_ratio),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.min_lexical_coverage > 1.0:
            raise ValueError("min_lexical_coverage must be at most 1")
        if self.min_sparse_score_ratio > 1.0:
            raise ValueError("min_sparse_score_ratio must be at most 1")
        if self.min_fusion_score_ratio > 1.0:
            raise ValueError("min_fusion_score_ratio must be at most 1")


@dataclass(frozen=True)
class RelevanceAssessment:
    status: RelevanceStatus
    accepted_hits: tuple[KnowledgeHit, ...]
    rejected_count: int
    top_score: float | None
    margin: float | None
    reason: str
    query_coverage: float | None = None
    sparse_top_score: float | None = None
    dense_top_score: float | None = None


def assess_relevance(
    query: str,
    hits: tuple[KnowledgeHit, ...],
    *,
    policy: RelevancePolicy | None = None,
) -> RelevanceAssessment:
    resolved = policy or RelevancePolicy()
    if not hits:
        return RelevanceAssessment("insufficient", (), 0, None, None, "no_hits")

    query_coverage = max((_lexical_coverage(query, hit) for hit in hits), default=0.0)
    sparse_top_score = max(
        (float(hit.sparse_score) for hit in hits if hit.sparse_score is not None),
        default=None,
    )
    dense_top_score = max(
        (float(hit.dense_score) for hit in hits if hit.dense_score is not None),
        default=None,
    )

    scored: list[tuple[KnowledgeHit, tuple[float, _SignalSpace] | None]] = []
    max_sparse_score = max(
        (float(hit.sparse_score) for hit in hits if hit.sparse_score is not None),
        default=None,
    )
    for hit in hits:
        signal = _admission_signal(query, hit, resolved, max_sparse_score=max_sparse_score)
        scored.append((hit, signal))

    explicit: list[tuple[KnowledgeHit, float, _SignalSpace]] = []
    for hit, signal in scored:
        if signal is not None:
            explicit.append((hit, signal[0], signal[1]))
    if not explicit:
        if any(hit.retrieval_backend != "unknown" for hit, _ in scored):
            return RelevanceAssessment(
                "insufficient",
                (),
                len(hits),
                None,
                None,
                "all_hits_below_floor",
                query_coverage,
                sparse_top_score,
                dense_top_score,
            )
        return RelevanceAssessment(
            "weak",
            hits,
            0,
            None,
            None,
            "legacy_unscored_hits",
            query_coverage,
            sparse_top_score,
            dense_top_score,
        )

    best_fusion_score = max(
        (
            float(hit.fusion_score)
            for hit, _, _ in explicit
            if hit.fusion_score is not None
        ),
        default=None,
    )
    accepted = tuple(
        hit
        for hit, _, _ in explicit
        if best_fusion_score is None
        or best_fusion_score <= 0
        or hit.fusion_score is None
        or float(hit.fusion_score) / best_fusion_score
        >= resolved.min_fusion_score_ratio
    )
    rejected_count = len(hits) - len(accepted)
    if not accepted:
        return RelevanceAssessment(
            "insufficient",
            (),
            len(hits),
            None,
            None,
            "all_hits_below_fusion_floor",
            query_coverage,
            sparse_top_score,
            dense_top_score,
        )
    top_score = float(explicit[0][1])
    margin = _same_backend_margin(explicit)

    status = "supported"
    reason = "relevance_floor"
    if margin is not None and margin < resolved.min_top_margin:
        status = "weak"
        reason = "ambiguous_top_margin"
    return RelevanceAssessment(
        status,
        accepted,
        rejected_count,
        top_score,
        margin,
        reason,
        query_coverage,
        sparse_top_score,
        dense_top_score,
    )


def _same_backend_margin(
    scored: list[tuple[KnowledgeHit, float, _SignalSpace]],
) -> float | None:
    """Compare adjacent signals only when they share a score space.

    Retrieval order is already the backend's ranking. A lexical coverage score,
    a normalized sparse score and a dense cosine are not interchangeable, so
    cross-backend subtraction would invent a confidence margin.
    """
    by_space: dict[_SignalSpace, list[float]] = {}
    for _, score, space in scored:
        by_space.setdefault(space, []).append(float(score))
    margins = [
        values[0] - values[1]
        for values in by_space.values()
        if len(values) >= 2
        for values in [sorted(values, reverse=True)]
    ]
    if not margins:
        return None
    return min(margins)


def _admission_signal(
    query: str,
    hit: KnowledgeHit,
    policy: RelevancePolicy,
    *,
    max_sparse_score: float | None,
) -> tuple[float, _SignalSpace] | None:
    if hit.retrieval_backend == "explicit_read":
        return 1.0, "explicit"

    lexical_coverage = _lexical_coverage(query, hit)
    lexical_signal = (
        lexical_coverage if lexical_coverage >= policy.min_lexical_coverage else None
    )
    sparse_signal = None
    if (
        max_sparse_score is not None
        and max_sparse_score > 0
        and hit.sparse_score is not None
    ):
        ratio = hit.sparse_score / max_sparse_score
        if ratio >= policy.min_sparse_score_ratio and (
            lexical_signal is not None or hit.fusion_score is not None
        ):
            sparse_signal = ratio
    if sparse_signal is not None and sparse_signal < policy.min_sparse_score_ratio:
        sparse_signal = None
    dense_signal = (
        hit.dense_score
        if hit.dense_score is not None and hit.dense_score >= policy.min_dense_score
        else None
    )

    if hit.retrieval_backend == "dense":
        return (dense_signal, "dense") if dense_signal is not None else None
    if hit.retrieval_backend == "hybrid":
        if hit.sparse_score is not None:
            if sparse_signal is not None:
                return sparse_signal, "sparse"
            if dense_signal is None:
                return None
        if dense_signal is not None:
            return dense_signal, "dense"
        return (lexical_signal, "lexical") if lexical_signal is not None else None
    if hit.retrieval_backend == "lexical_custom":
        # Cached sparse hits have a backend-local score. Coverage alone is not
        # enough: generic words such as "hệ thống" can appear in unrelated
        # notes. Keep the lexical fallback only for legacy/custom hits that do
        # not expose a sparse score at all.
        if hit.sparse_score is not None:
            if sparse_signal is not None:
                return sparse_signal, "sparse"
            return None
        return (lexical_signal, "lexical") if lexical_signal is not None else None
    if hit.retrieval_backend == "unknown":
        return None if not query.strip() else None
    if dense_signal is not None:
        return dense_signal, "dense"
    if lexical_signal is not None:
        return lexical_signal, "lexical"
    return None


def _lexical_coverage(query: str, hit: KnowledgeHit) -> float:
    query_terms = set(tokenize_terms(query))
    if not query_terms:
        return 0.0
    evidence_terms = set(
        tokenize_terms(
            " ".join(
                (
                    hit.document.title,
                    hit.document.path,
                    hit.snippet,
                )
            )
        )
    )
    return len(query_terms & evidence_terms) / len(query_terms)


__all__ = ["RelevanceAssessment", "RelevancePolicy", "RelevanceStatus", "assess_relevance"]
