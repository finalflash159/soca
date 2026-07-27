from __future__ import annotations

import pytest

from soca.knowledge.retriever import RankedHit
from soca.knowledge.retrievers.rrf import reciprocal_rank_fusion


def _hit(chunk_id: str, rank: int, score: float = 1.0) -> RankedHit:
    return RankedHit(chunk_id=chunk_id, rank=rank, score=score)


def test_rrf_combines_rank_lists_with_exact_reciprocal_math() -> None:
    fused = reciprocal_rank_fusion(
        [
            [_hit("a", 1), _hit("b", 2)],
            [_hit("b", 1), _hit("c", 2)],
        ],
        k=10,
    )

    assert fused[0][0] == "b"
    assert fused[0][1] == pytest.approx(1 / 11 + 1 / 12)
    assert fused[1] == ("a", pytest.approx(1 / 11))
    assert fused[2] == ("c", pytest.approx(1 / 12))


def test_rrf_uses_chunk_id_for_deterministic_ties_and_accepts_empty_lists() -> None:
    fused = reciprocal_rank_fusion(
        [[], [_hit("z", 1)], [_hit("a", 1)]],
        k=60,
    )

    assert fused == (("a", pytest.approx(1 / 61)), ("z", pytest.approx(1 / 61)))


def test_rrf_rejects_duplicate_hits_from_one_retriever() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        reciprocal_rank_fusion([[_hit("a", 1), _hit("a", 2)]])


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_rrf_rejects_invalid_k(k: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        reciprocal_rank_fusion([], k=k)  # type: ignore[arg-type]
