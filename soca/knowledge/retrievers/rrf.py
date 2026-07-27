from __future__ import annotations

from collections.abc import Sequence

from soca.knowledge.retriever import RankedHit


def reciprocal_rank_fusion(
    rank_lists: Sequence[Sequence[RankedHit]],
    *,
    k: int = 60,
) -> tuple[tuple[str, float], ...]:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")

    fused: dict[str, float] = {}
    for hits in rank_lists:
        seen: set[str] = set()
        for hit in hits:
            if hit.chunk_id in seen:
                raise ValueError("one retriever returned a duplicate chunk id")
            seen.add(hit.chunk_id)
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (k + hit.rank)

    return tuple(sorted(fused.items(), key=lambda item: (-item[1], item[0])))
