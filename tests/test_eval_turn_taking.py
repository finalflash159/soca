from __future__ import annotations

import numpy as np
import pytest

from eval.eval_turn_taking import choose_floor, evaluate_floor_sweep
from eval.scenarios_turn_taking import make_clean


class _EnergyVad:
    def __call__(self, frame: np.ndarray, sample_rate: int) -> float:
        return 1.0 if np.max(np.abs(frame)) > 0.1 else 0.0


class _CompleteTurn:
    def p_still_speaking(self, audio_window: np.ndarray) -> float:
        return 0.0


def test_floor_sweep_runs_paired_scenarios_with_each_config() -> None:
    scenario = make_clean(
        "fixture",
        np.full(16_000, 0.2, dtype=np.float32),
        trailing_ms=3500,
    )
    resets = 0

    def reset() -> None:
        nonlocal resets
        resets += 1

    report = evaluate_floor_sweep(
        [scenario],
        vad=_EnergyVad(),
        vad_reset=reset,
        turn_detector=_CompleteTurn(),
        floors_ms=(1000, 1400),
    )

    assert resets == 2
    assert set(report) == {"1000", "1400"}
    assert report["1000"]["median_over_wait_ms"] == pytest.approx(1024.0, abs=32)
    assert report["1400"]["median_over_wait_ms"] == pytest.approx(1408.0, abs=32)


def test_choose_floor_selects_lowest_latency_candidate_that_meets_both_gates() -> None:
    report = {
        "1000": {
            "cut_in_rate": 0.04,
            "premature_close_rate": 0.10,
            "median_over_wait_ms": 1050.0,
        },
        "1200": {
            "cut_in_rate": 0.05,
            "premature_close_rate": 0.05,
            "median_over_wait_ms": 1250.0,
        },
        "1400": {
            "cut_in_rate": 0.01,
            "premature_close_rate": 0.03,
            "median_over_wait_ms": 1450.0,
        },
    }

    assert choose_floor(report) == 1200


def test_choose_floor_has_no_implicit_fallback_when_every_candidate_fails() -> None:
    report = {
        "1800": {
            "cut_in_rate": 0.01,
            "premature_close_rate": 0.06,
            "median_over_wait_ms": 1824.0,
        }
    }

    assert choose_floor(report) is None
