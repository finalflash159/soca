"""Tests for turn-taking scenario synthesis + metrics (P3.1 Pha B/C, Tier 2).

Pure array/frame math — no FLEURS files, no models.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.conversation_metrics import (
    SCENARIO_CLEAN,
    SCENARIO_MID_PAUSE,
    TurnOutcome,
    turn_taking_report,
)
from eval.scenarios_turn_taking import make_clean, make_mid_pause

_SR = 16000


def _tone(ms: float) -> np.ndarray:
    return np.full(int(ms / 1000 * _SR), 0.2, dtype=np.float32)


def test_make_clean_marks_turn_end_at_utterance_end() -> None:
    scenario = make_clean("u1", _tone(1000), trailing_ms=3500)

    assert scenario.scenario_type == SCENARIO_CLEAN
    assert scenario.true_end_ms == pytest.approx(1000.0, abs=1.0)
    assert len(scenario.near) == pytest.approx((1000 + 3500) / 1000 * _SR, abs=2)
    assert scenario.pause_end_ms is None


def test_make_mid_pause_places_pause_and_true_end() -> None:
    scenario = make_mid_pause("u1", _tone(2000), pause_ms=800, split_frac=0.5)

    assert scenario.scenario_type == SCENARIO_MID_PAUSE
    # 2000ms utterance split 50/50 → pause starts at 1000ms, lasts 800ms.
    assert scenario.pause_start_ms == pytest.approx(1000.0, abs=1.0)
    assert scenario.pause_ms == pytest.approx(800.0)
    assert scenario.pause_end_ms == pytest.approx(1800.0, abs=1.0)
    # true end = first(1000) + pause(800) + second(1000) = 2800ms.
    assert scenario.true_end_ms == pytest.approx(2800.0, abs=1.0)


def test_metrics_flag_cut_in_and_over_wait() -> None:
    # eager policy stops inside the pause (cut-in); patient waits past the real end.
    outcomes = [
        # clean: eager over-waits 700ms, patient 1000ms (both correct closes)
        TurnOutcome(SCENARIO_CLEAN, "fixed", True, stop_ms=1700, true_end_ms=1000),
        TurnOutcome(SCENARIO_CLEAN, "p_based", True, stop_ms=2000, true_end_ms=1000),
        # mid_pause @ [1000,1800]: eager stops at 1700 (inside → cut-in); patient at 3500
        TurnOutcome(SCENARIO_MID_PAUSE, "fixed", True, stop_ms=1700, true_end_ms=2800, pause_end_ms=1800),
        TurnOutcome(SCENARIO_MID_PAUSE, "p_based", True, stop_ms=3500, true_end_ms=2800, pause_end_ms=1800),
    ]
    report = turn_taking_report(outcomes)

    assert report["fixed"]["cut_in_rate"] == pytest.approx(1.0)
    assert report["p_based"]["cut_in_rate"] == pytest.approx(0.0)
    assert report["fixed"]["median_over_wait_ms"] == pytest.approx(700.0)
    assert report["p_based"]["median_over_wait_ms"] == pytest.approx(1000.0)


def test_premature_close_excluded_from_over_wait() -> None:
    # A clean turn closed before the real end is a premature error, not a negative
    # over-wait — it must not enter the over-wait distribution.
    outcomes = [
        TurnOutcome(SCENARIO_CLEAN, "fixed", True, stop_ms=2400, true_end_ms=11000),  # early
        TurnOutcome(SCENARIO_CLEAN, "fixed", True, stop_ms=11700, true_end_ms=11000),  # correct
    ]
    report = turn_taking_report(outcomes)

    assert report["fixed"]["premature_close_rate"] == pytest.approx(0.5)
    assert report["fixed"]["median_over_wait_ms"] == pytest.approx(700.0)  # only the correct one
    assert report["fixed"]["n_correct_close"] == 1


def test_metrics_empty_is_safe() -> None:
    assert turn_taking_report([]) == {}
