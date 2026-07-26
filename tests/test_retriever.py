from __future__ import annotations

import math

import pytest

from soca.knowledge.retriever import RankedHit


def test_ranked_hit_is_immutable_and_preserves_valid_values() -> None:
    hit = RankedHit(chunk_id="wiki/study/bayes.md#2", rank=1, score=0.75)

    assert hit.chunk_id == "wiki/study/bayes.md#2"
    assert hit.rank == 1
    assert hit.score == 0.75
    with pytest.raises(AttributeError):
        hit.rank = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("chunk_id", "rank", "score", "message"),
    [
        ("", 1, 0.1, "chunk_id"),
        ("   ", 1, 0.1, "chunk_id"),
        ("chunk", 0, 0.1, "rank"),
        ("chunk", -1, 0.1, "rank"),
        ("chunk", True, 0.1, "rank"),
        ("chunk", 1, math.inf, "score"),
        ("chunk", 1, math.nan, "score"),
    ],
)
def test_ranked_hit_rejects_invalid_ranking_values(
    chunk_id: str,
    rank: int,
    score: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        RankedHit(chunk_id=chunk_id, rank=rank, score=score)
