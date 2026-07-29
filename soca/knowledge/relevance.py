from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from soca.knowledge.base import KnowledgeHit
from soca.knowledge.markdown_vault import tokenize_terms

RelevanceStatus = Literal["supported", "weak", "insufficient"]


@dataclass(frozen=True)
class RelevancePolicy:
    """Calibratable admission policy for retrieved evidence.

    Scores are backend-local signals. A dense cosine threshold is never
    compared with a lexical or fusion score; each backend contributes its own
    signal and the context builder records the resulting decision.
    """

    min_lexical_coverage: float = 0.60
    min_sparse_score_ratio: float = 0.75
    min_dense_score: float = 0.55
    min_top_margin: float = 0.05

    def __post_init__(self) -> None:
        for name, value in (
            ("min_lexical_coverage", self.min_lexical_coverage),
            ("min_sparse_score_ratio", self.min_sparse_score_ratio),
            ("min_dense_score", self.min_dense_score),
            ("min_top_margin", self.min_top_margin),
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


@dataclass(frozen=True)
class RelevanceAssessment:
    status: RelevanceStatus
    accepted_hits: tuple[KnowledgeHit, ...]
    rejected_count: int
    top_score: float | None
    margin: float | None
    reason: str


def assess_relevance(
    query: str,
    hits: tuple[KnowledgeHit, ...],
    *,
    policy: RelevancePolicy | None = None,
) -> RelevanceAssessment:
    resolved = policy or RelevancePolicy()
    if not hits:
        return RelevanceAssessment("insufficient", (), 0, None, None, "no_hits")

    scored: list[tuple[KnowledgeHit, float | None]] = []
    max_sparse_score = max(
        (float(hit.sparse_score) for hit in hits if hit.sparse_score is not None),
        default=None,
    )
    for hit in hits:
        signal = _admission_signal(query, hit, resolved, max_sparse_score=max_sparse_score)
        scored.append((hit, signal))

    explicit = [item for item in scored if item[1] is not None]
    if not explicit:
        if any(hit.retrieval_backend != "unknown" for hit, _ in scored):
            return RelevanceAssessment(
                "insufficient",
                (),
                len(hits),
                None,
                None,
                "all_hits_below_floor",
            )
        return RelevanceAssessment(
            "weak",
            hits,
            0,
            None,
            None,
            "legacy_unscored_hits",
        )

    explicit.sort(key=lambda item: (-float(item[1]), item[0].document.path, item[0].document.id))
    accepted = tuple(hit for hit, signal in explicit if signal is not None)
    rejected_count = len(hits) - len(accepted)
    scores = [float(signal) for _, signal in explicit]
    top_score = scores[0]
    margin = top_score - scores[1] if len(scores) > 1 else None
    if not accepted:
        return RelevanceAssessment(
            "insufficient",
            (),
            len(hits),
            top_score,
            margin,
            "all_hits_below_floor",
        )

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
    )


def _admission_signal(
    query: str,
    hit: KnowledgeHit,
    policy: RelevancePolicy,
    *,
    max_sparse_score: float | None,
) -> float | None:
    if hit.retrieval_backend == "explicit_read":
        return 1.0

    lexical_coverage = _lexical_coverage(query, hit)
    lexical_signal = (
        lexical_coverage if lexical_coverage >= policy.min_lexical_coverage else None
    )
    sparse_signal = (
        hit.sparse_score / max_sparse_score
        if (
            lexical_coverage > 0.0
            and max_sparse_score is not None
            and max_sparse_score > 0
            and hit.sparse_score is not None
        )
        else None
    )
    if sparse_signal is not None and sparse_signal < policy.min_sparse_score_ratio:
        sparse_signal = None
    dense_signal = (
        hit.dense_score
        if hit.dense_score is not None and hit.dense_score >= policy.min_dense_score
        else None
    )

    if hit.retrieval_backend == "dense":
        return dense_signal
    if hit.retrieval_backend == "hybrid":
        return max(
            (value for value in (lexical_signal, sparse_signal, dense_signal) if value is not None),
            default=None,
        )
    if hit.retrieval_backend == "lexical_custom":
        return max(
            (value for value in (lexical_signal, sparse_signal) if value is not None),
            default=None,
        )
    if hit.retrieval_backend == "unknown":
        return None if not query.strip() else None
    return max(
        (value for value in (lexical_signal, dense_signal) if value is not None),
        default=None,
    )


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
