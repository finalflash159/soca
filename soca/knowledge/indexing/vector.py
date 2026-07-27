from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from soca.knowledge.indexing.identity import SearchBackendFingerprint
from soca.knowledge.retriever import RankedHit


def stable_exact_top_k(
    scores: np.ndarray,
    stable_ids: tuple[str, ...],
    *,
    limit: int,
) -> tuple[int, ...]:
    """Return exact top-k with deterministic score-desc/id-asc ordering.

    ``argpartition`` alone is not stable when several rows share the kth score.
    We therefore select the complete kth boundary and only sort that (usually
    small) candidate set. Non-finite values are rejected instead of silently
    becoming arbitrary nearest neighbours.
    """
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 1 or len(values) != len(stable_ids) or not np.isfinite(values).all():
        raise ValueError("dense scores must be a finite one-dimensional vector")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be positive")
    if not len(values):
        return ()
    count = min(limit, len(values))
    if count == len(values):
        candidates = range(len(values))
    else:
        partition = np.argpartition(-values, count - 1)[:count]
        threshold = float(np.min(values[partition]))
        candidates = np.flatnonzero(values >= threshold).tolist()
    ordered = sorted(candidates, key=lambda index: (-float(values[index]), stable_ids[index]))
    return tuple(ordered[:count])


@dataclass(frozen=True)
class NumpyExactVectorIndex:
    vectors: np.ndarray
    stable_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        matrix = np.asarray(self.vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(self.stable_ids) or matrix.shape[1] < 1:
            raise ValueError("vector matrix shape is invalid")
        if not np.isfinite(matrix).all():
            raise ValueError("vector matrix must be finite")
        if any(not value.strip() for value in self.stable_ids):
            raise ValueError("stable ids must be non-empty")
        if len(set(self.stable_ids)) != len(self.stable_ids):
            raise ValueError("stable ids must be unique")
        norms = np.linalg.norm(matrix, axis=1)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
            raise ValueError("vector matrix contains a zero-norm row")
        readonly = np.asarray(matrix, dtype=np.float32)
        readonly.setflags(write=False)
        object.__setattr__(self, "vectors", readonly)

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def fingerprint(self) -> SearchBackendFingerprint:
        return SearchBackendFingerprint(library_version=f"numpy-{np.__version__}")

    def search(self, query: np.ndarray, *, limit: int) -> tuple[RankedHit, ...]:
        vector = np.asarray(query, dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != self.dimension or not np.isfinite(vector).all():
            raise ValueError("query vector shape is invalid")
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError("query vector must have non-zero norm")
        normalized = vector / norm
        scores = self.vectors @ normalized
        order = stable_exact_top_k(scores, self.stable_ids, limit=limit)
        return tuple(
            RankedHit(
                chunk_id=self.stable_ids[index],
                rank=rank,
                score=float(scores[index]),
            )
            for rank, index in enumerate(order, start=1)
        )
