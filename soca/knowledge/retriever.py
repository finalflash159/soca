from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RankedHit:
    chunk_id: str
    rank: int
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise ValueError("chunk_id must be a non-empty string")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("rank must be a positive integer")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise ValueError("score must be a finite number")
        if not math.isfinite(float(self.score)):
            raise ValueError("score must be a finite number")


class Retriever(Protocol):
    @property
    def available(self) -> bool: ...

    def rank(self, query: str, *, limit: int) -> list[RankedHit]: ...
