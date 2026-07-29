from __future__ import annotations

import pytest

from eval.retrieval_analysis import MeasurementPair, holm_adjust, paired_bootstrap


def test_paired_bootstrap_is_deterministic_and_reports_direction() -> None:
    pairs = tuple(
        MeasurementPair(str(index), baseline=0.2, candidate=0.8)
        for index in range(20)
    )

    result = paired_bootstrap(pairs, samples=2_000, seed=42)

    assert result.delta == pytest.approx(0.6)
    assert result.ci_low > 0
    assert result.ci_high > 0
    assert result.p_value < 0.01


def test_holm_adjust_is_monotonic_in_sorted_order() -> None:
    adjusted = holm_adjust({"a": 0.001, "b": 0.02, "c": 0.2})

    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
    assert all(0 <= value <= 1 for value in adjusted.values())
