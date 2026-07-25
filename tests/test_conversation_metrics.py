"""Tests for the Tier 1 barge-in metrics (P3.1 Pha C). Pure, no audio."""

from __future__ import annotations

import pytest

from eval.conversation_metrics import (
    CONDITION_DOUBLE_TALK,
    CONDITION_ECHO_ONLY,
    BargeInOutcome,
    barge_in_report,
)


def echo(interrupted: bool, *, moving: bool = False) -> BargeInOutcome:
    return BargeInOutcome(
        condition=CONDITION_ECHO_ONLY,
        expected_interrupt=False,
        interrupted=interrupted,
        with_movement=moving,
    )


def dbl(interrupted: bool, *, fire_ms: float | None = None, moving: bool = False) -> BargeInOutcome:
    return BargeInOutcome(
        condition=CONDITION_DOUBLE_TALK,
        expected_interrupt=True,
        interrupted=interrupted,
        interrupt_ms=fire_ms,
        with_movement=moving,
    )


def test_false_interrupt_and_detection_rates() -> None:
    outcomes = [
        echo(False), echo(False), echo(False), echo(True),  # 1/4 false-interrupt
        dbl(True, fire_ms=500), dbl(True, fire_ms=700), dbl(False),  # 2/3 detection
    ]
    report = barge_in_report(outcomes)

    assert report.n_total == 7
    assert report.n_echo_only == 4 and report.n_double_talk == 3
    assert report.false_interrupt_rate == pytest.approx(0.25)
    assert report.detection_rate == pytest.approx(2 / 3)
    assert report.missed_rate == pytest.approx(1 / 3)
    assert report.median_fire_ms == pytest.approx(600.0)  # median of 500, 700


def test_median_fire_ms_ignores_missed_and_echo_fires() -> None:
    # Only double_talk *detections* contribute to the fire-time distribution.
    outcomes = [
        echo(True),  # a false fire, must not enter fire_ms
        dbl(True, fire_ms=400),
        dbl(False, fire_ms=None),
    ]
    report = barge_in_report(outcomes)

    assert report.median_fire_ms == pytest.approx(400.0)


def test_movement_split_separates_static_and_moving() -> None:
    outcomes = [
        echo(False, moving=False), echo(True, moving=True),  # moving echo false-fires
        dbl(True, fire_ms=500, moving=False), dbl(False, moving=True),  # moving misses
    ]
    report = barge_in_report(outcomes)

    assert report.by_movement["static"]["false_interrupt_rate"] == pytest.approx(0.0)
    assert report.by_movement["static"]["detection_rate"] == pytest.approx(1.0)
    assert report.by_movement["moving"]["false_interrupt_rate"] == pytest.approx(1.0)
    assert report.by_movement["moving"]["detection_rate"] == pytest.approx(0.0)


def test_empty_outcomes_do_not_crash() -> None:
    report = barge_in_report([])

    assert report.n_total == 0
    assert report.false_interrupt_rate == pytest.approx(0.0)
    assert report.detection_rate == pytest.approx(0.0)
    assert report.missed_rate == pytest.approx(0.0)
    assert report.median_fire_ms is None
    assert report.by_movement == {}


def test_all_echo_only_reports_zero_detection_denominator() -> None:
    report = barge_in_report([echo(False), echo(True)])

    assert report.false_interrupt_rate == pytest.approx(0.5)
    assert report.detection_rate == pytest.approx(0.0)  # no double_talk → 0, not a crash
    assert report.missed_rate == pytest.approx(0.0)
