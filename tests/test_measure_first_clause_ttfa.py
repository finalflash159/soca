from __future__ import annotations

from eval.measure_first_clause_ttfa import summarize_measurements


def test_summary_uses_nearest_rank_and_declares_help_threshold() -> None:
    summary = summarize_measurements([0.0, 10.0, 20.0, 30.0], helped_threshold_ms=5.0)

    assert summary == {
        "count": 4,
        "p50_ms": 15.0,
        "p95_ms": 30.0,
        "min_ms": 0.0,
        "max_ms": 30.0,
        "helped_threshold_ms": 5.0,
        "helped_count": 3,
    }


def test_summary_is_explicit_when_no_measurement_is_available() -> None:
    assert summarize_measurements([], helped_threshold_ms=5.0) == {
        "count": 0,
        "p50_ms": None,
        "p95_ms": None,
        "min_ms": None,
        "max_ms": None,
        "helped_threshold_ms": 5.0,
        "helped_count": 0,
    }
