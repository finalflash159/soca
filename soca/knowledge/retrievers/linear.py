from __future__ import annotations

from collections.abc import Sequence

from soca.knowledge.retriever import RankedHit


def _normalized_scores(hits: Sequence[RankedHit]) -> dict[str, float]:
    if not hits:
        return {}
    values = tuple(hit.score for hit in hits)
    low = min(values)
    span = max(values) - low
    if span <= 1e-12:
        return {hit.chunk_id: 0.0 for hit in hits}
    return {hit.chunk_id: (hit.score - low) / span for hit in hits}


def linear_score_fusion(
    sparse_hits: Sequence[RankedHit],
    dense_hits: Sequence[RankedHit],
    *,
    dense_weight: float,
) -> tuple[tuple[str, float], ...]:
    if isinstance(dense_weight, bool) or not isinstance(dense_weight, (int, float)):
        raise ValueError("dense_weight must be a number")
    if not 0.0 <= float(dense_weight) <= 1.0:
        raise ValueError("dense_weight must be in [0, 1]")
    sparse = _normalized_scores(sparse_hits)
    dense = _normalized_scores(dense_hits)
    identifiers = set(sparse) | set(dense)
    weight = float(dense_weight)
    return tuple(
        sorted(
            (
                (
                    identifier,
                    (1.0 - weight) * sparse.get(identifier, 0.0)
                    + weight * dense.get(identifier, 0.0),
                )
                for identifier in identifiers
            ),
            key=lambda item: (-item[1], item[0]),
        )
    )
