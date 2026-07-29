from __future__ import annotations

from eval.retrieval_resource_probe import _percentile


def test_resource_probe_percentile_uses_nearest_rank() -> None:
    assert _percentile([10.0, 1.0, 3.0, 2.0], 0.50) == 2.0
    assert _percentile([10.0, 1.0, 3.0, 2.0], 0.95) == 10.0
