"""Wiring test for the Tier 1 driver (P3.1 Pha C).

Exercises load_pair → BargeInDecider → outcome → report over tiny synthetic flac
pairs, with fake AEC/VAD injected — no WebRTC, no Silero.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from eval.aec_challenge import discover_pairs
from eval.conversation_metrics import barge_in_report
from eval.eval_conversation import evaluate_scenarios

_SR = 16000
_FRAME = 512


class _SubtractAec:
    def process(self, near: np.ndarray, far: np.ndarray) -> np.ndarray:
        return near - far


class _EnergyVad:
    def __call__(self, frame: np.ndarray, sample_rate: int) -> float:
        return 1.0 if float(np.sqrt(np.mean(frame**2))) > 0.05 else 0.0


def _write(path: Path, audio: np.ndarray) -> None:
    sf.write(path, audio.astype(np.float32), _SR)


def test_driver_detects_double_talk_and_ignores_echo(tmp_path: Path) -> None:
    n = 20 * _FRAME
    far = np.full(n, 0.3, dtype=np.float32)

    # echo_only: mic == far → ideal AEC cancels to silence → must NOT interrupt.
    _write(tmp_path / "e1_farend_singletalk_mic.flac", far.copy())
    _write(tmp_path / "e1_farend_singletalk_lpb.flac", far.copy())

    # double_talk: user speaks over echo from frame 5 → clean = user → interrupt.
    mic = far.copy()
    mic[5 * _FRAME :] += 0.2
    _write(tmp_path / "d1_doubletalk_mic.flac", mic)
    _write(tmp_path / "d1_doubletalk_lpb.flac", far.copy())

    scenarios = discover_pairs(tmp_path)
    outcomes = evaluate_scenarios(
        scenarios,
        aec_factory=_SubtractAec,
        vad=_EnergyVad(),
        vad_reset=lambda: None,
        sustained_ms=400.0,
        vad_threshold=0.7,
    )
    report = barge_in_report(outcomes)

    assert report.n_echo_only == 1 and report.n_double_talk == 1
    assert report.false_interrupt_rate == pytest.approx(0.0)  # echo ignored
    assert report.detection_rate == pytest.approx(1.0)  # user caught
    assert report.median_fire_ms is not None
