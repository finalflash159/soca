"""Tests for synthesized VN barge-in scenarios + latency metrics (P3.1 Tier 1 synth).

Synthetic arrays + fake AEC/VAD — no FLEURS, no RIR files, no models.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.conversation_metrics import (
    KIND_BACKCHANNEL,
    KIND_BARGE_IN,
    KIND_ECHO_ONLY,
    SynthBargeOutcome,
    synth_barge_report,
)
from eval.eval_barge_in_synth import evaluate_synth
from eval.scenarios_barge_in_synth import (
    make_barge_in,
    make_echo_only,
    synth_echo,
)

_SR = 16000
_FRAME = 512


class _SubtractAec:
    def process(self, near: np.ndarray, far: np.ndarray) -> np.ndarray:
        return near - far


class _EnergyVad:
    def __call__(self, frame: np.ndarray, sample_rate: int) -> float:
        return 1.0 if float(np.sqrt(np.mean(frame**2))) > 0.05 else 0.0


def test_synth_echo_truncates_to_far_length() -> None:
    far = np.ones(10 * _FRAME, dtype=np.float32)
    rir = np.array([1.0, 0.5, 0.25], dtype=np.float32)

    echo = synth_echo(far, rir, alpha=0.5)

    assert len(echo) == len(far)


def test_make_barge_in_places_user_at_onset() -> None:
    far = np.full(30 * _FRAME, 0.3, dtype=np.float32)
    rir = np.array([1.0], dtype=np.float32)  # identity RIR + alpha=1 → echo == far
    user = np.full(10 * _FRAME, 0.2, dtype=np.float32)

    scenario = make_barge_in(far, rir, user, onset_ms=500.0, alpha=1.0)

    assert scenario.kind == KIND_BARGE_IN
    assert scenario.expected_interrupt and scenario.onset_ms == pytest.approx(500.0)
    assert len(scenario.far) == len(scenario.near)
    # ideal AEC (near−far) leaves the user; energy rises after the onset sample.
    onset = int(500 / 1000 * _SR)
    clean = scenario.near - scenario.far
    assert float(np.abs(clean[:onset]).mean()) < 1e-6  # silence before onset
    assert float(np.abs(clean[onset + _FRAME]).mean()) > 0.1  # user after onset


def test_latency_is_fire_minus_onset_and_excludes_prefire() -> None:
    outcomes = [
        SynthBargeOutcome(KIND_BARGE_IN, True, True, onset_ms=1000, interrupt_ms=1400),  # 400ms
        SynthBargeOutcome(KIND_BARGE_IN, True, True, onset_ms=1000, interrupt_ms=1600),  # 600ms
        SynthBargeOutcome(KIND_BARGE_IN, True, True, onset_ms=1000, interrupt_ms=800),  # pre-fire!
        SynthBargeOutcome(KIND_ECHO_ONLY, False, False),
        SynthBargeOutcome(KIND_BACKCHANNEL, False, True),  # a backchannel that wrongly fired
    ]
    report = synth_barge_report(outcomes)

    assert report.false_interrupt_rate == pytest.approx(0.0)
    assert report.backchannel_fire_rate == pytest.approx(1.0)
    # 2 of 3 barge_ins detected (the pre-fire is excluded from detection + latency).
    assert report.detection_rate == pytest.approx(2 / 3)
    assert report.median_latency_ms == pytest.approx(500.0)  # median of 400, 600


def test_evaluate_synth_end_to_end_with_fakes() -> None:
    far = np.full(30 * _FRAME, 0.3, dtype=np.float32)
    rir = np.array([1.0], dtype=np.float32)
    user = np.full(15 * _FRAME, 0.2, dtype=np.float32)
    scenarios = [
        make_echo_only(far, rir, alpha=1.0),  # echo == far → SubtractAec → 0 → no fire
        make_barge_in(far, rir, user, onset_ms=500.0, alpha=1.0),  # user → fire
    ]

    outcomes = evaluate_synth(
        scenarios,
        aec_factory=_SubtractAec,
        vad=_EnergyVad(),
        vad_reset=lambda: None,
        sustained_ms=400.0,
        vad_threshold=0.7,
    )
    report = synth_barge_report(outcomes)

    assert report.false_interrupt_rate == pytest.approx(0.0)
    assert report.detection_rate == pytest.approx(1.0)
    assert report.median_latency_ms is not None and report.median_latency_ms > 0
