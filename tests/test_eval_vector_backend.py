from __future__ import annotations

import numpy as np
import pytest

from eval.eval_vector_backend import (
    memory_preflight,
    stable_exact_top_k,
)


def test_stable_exact_top_k_matches_score_then_row_order() -> None:
    scores = np.asarray([0.8, 0.9, 0.8, 0.8, 0.1], dtype=np.float32)

    assert stable_exact_top_k(scores, limit=3).tolist() == [1, 0, 2]


def test_stable_exact_top_k_handles_limit_above_row_count() -> None:
    scores = np.asarray([0.1, 0.3, 0.2], dtype=np.float32)

    assert stable_exact_top_k(scores, limit=10).tolist() == [1, 2, 0]


def test_stable_exact_top_k_rejects_non_finite_scores() -> None:
    scores = np.asarray([0.1, np.nan], dtype=np.float32)

    with pytest.raises(ValueError, match="finite"):
        stable_exact_top_k(scores, limit=1)


def test_memory_preflight_rejects_unsafe_allocation() -> None:
    with pytest.raises(MemoryError, match="max-memory-mib"):
        memory_preflight(
            size=1_000_000,
            dimension=1024,
            query_count=40,
            backends=("faiss-hnsw",),
            max_memory_mib=1024,
        )
