"""Tests for the conversational replay deciders (P3.1 Pha A).

Pure synthetic frames + injected AEC/VAD/turn-detector fakes — no sounddevice,
no Silero, no ONNX. Fast in CI. Every timing assertion is exact because replay
time is frame index, not wall clock.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.barge_in_replay import (
    POLICY_FIXED,
    POLICY_P_BASED,
    BargeInDecider,
    TurnEndpointDecider,
)
from soca.core.turn_taking import VAD_FRAME_MS, VAD_FRAME_SAMPLES

_BARGE_FRAME = 512  # 32ms @16k, DuplexAecSink default


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class _PassThroughAec:
    def process(self, near: np.ndarray, far: np.ndarray) -> np.ndarray:
        return near


class _SubtractAec:
    """Ideal echo canceller: perfectly removes the far reference from the mic."""

    def process(self, near: np.ndarray, far: np.ndarray) -> np.ndarray:
        return near - far


class _ScriptedVad:
    """Return 1.0 for the frame indices in ``speech``, else 0.0 (call-ordered)."""

    def __init__(self, speech: set[int]) -> None:
        self._speech = set(speech)
        self.calls = 0

    def __call__(self, frame: np.ndarray, sample_rate: int) -> float:
        idx = self.calls
        self.calls += 1
        return 1.0 if idx in self._speech else 0.0


class _EnergyVad:
    """Speech iff the (post-AEC) frame carries energy above a small floor."""

    def __init__(self, floor: float = 0.05) -> None:
        self.floor = floor

    def __call__(self, frame: np.ndarray, sample_rate: int) -> float:
        return 1.0 if float(np.sqrt(np.mean(frame**2))) > self.floor else 0.0


class _FixedTurn:
    def __init__(self, p: float) -> None:
        self.p = p

    def p_still_speaking(self, audio_window: np.ndarray) -> float:
        return self.p


def _zeros(n_frames: int, frame: int = _BARGE_FRAME) -> np.ndarray:
    return np.zeros(n_frames * frame, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Tier 1 — barge-in
# --------------------------------------------------------------------------- #


def test_barge_in_fires_after_sustained_speech() -> None:
    # sustained 400ms / 32ms block => run reaches 400 on the 13th speech frame (idx 12).
    decider = BargeInDecider(aec=_PassThroughAec(), vad=_ScriptedVad(set(range(20))))
    result = decider.run(_zeros(20), _zeros(20))

    assert result.interrupted
    assert result.interrupt_frame == 12
    assert result.interrupt_ms == pytest.approx(416.0)  # (12+1)*32
    assert result.frames_processed == 13  # stops at the firing frame
    assert result.speech_frames == 13


def test_barge_in_does_not_fire_below_sustained_threshold() -> None:
    # 10 speech frames = 320ms < 400ms sustained => never fires.
    decider = BargeInDecider(aec=_PassThroughAec(), vad=_ScriptedVad(set(range(10))))
    result = decider.run(_zeros(25), _zeros(25))

    assert not result.interrupted
    assert result.interrupt_frame is None
    assert result.max_run_ms == pytest.approx(320.0)
    assert result.frames_processed == 25


def test_barge_in_run_resets_on_a_silence_gap() -> None:
    # Two 10-frame runs split by one silent frame: neither reaches 400ms.
    speech = set(range(10)) | set(range(11, 21))
    decider = BargeInDecider(aec=_PassThroughAec(), vad=_ScriptedVad(speech))
    result = decider.run(_zeros(22), _zeros(22))

    assert not result.interrupted
    assert result.max_run_ms == pytest.approx(320.0)


def test_barge_in_ignores_pure_echo_but_catches_real_user() -> None:
    # Echo = far played back into the mic; ideal AEC cancels it to silence.
    far = np.full(20 * _BARGE_FRAME, 0.3, dtype=np.float32)
    energy_decider = lambda near: BargeInDecider(  # noqa: E731
        aec=_SubtractAec(), vad=_EnergyVad()
    ).run(far, near)

    echo_only = energy_decider(far.copy())  # near == far → clean ≈ 0
    assert not echo_only.interrupted
    assert echo_only.speech_frames == 0

    # User speaks over the echo from frame 5 on → clean = user → sustained → fire.
    with_user = far.copy()
    with_user[5 * _BARGE_FRAME :] += 0.2
    fired = energy_decider(with_user)
    assert fired.interrupted
    assert fired.interrupt_frame == 17  # onset at 5, +12 frames to reach 400ms


def test_barge_in_rejects_misaligned_buffers() -> None:
    decider = BargeInDecider(aec=_PassThroughAec(), vad=_ScriptedVad(set()))
    with pytest.raises(ValueError, match="same length"):
        decider.run(_zeros(5), _zeros(6))


# --------------------------------------------------------------------------- #
# Tier 2 — turn-taking
# --------------------------------------------------------------------------- #


def _speech_then_silence(n_speech: int, n_silence: int) -> tuple[np.ndarray, set[int]]:
    total = n_speech + n_silence
    return _zeros(total, VAD_FRAME_SAMPLES), set(range(n_speech))


def test_fixed_policy_stops_after_constant_silence() -> None:
    near, speech = _speech_then_silence(5, 25)
    decider = TurnEndpointDecider(vad=_ScriptedVad(speech), policy=POLICY_FIXED)
    result = decider.run(near)

    assert result.stopped
    assert result.policy == POLICY_FIXED
    assert result.required_silence_ms == pytest.approx(700.0)  # endpoint_silence_ms
    # 700ms / 32ms => 22 silence frames; silence starts at index 5 → stop at 26.
    assert result.stop_frame == 26
    assert result.trailing_silence_ms == pytest.approx(704.0)


def test_p_based_waits_longer_when_model_says_still_speaking() -> None:
    near, speech = _speech_then_silence(5, 100)
    eager = TurnEndpointDecider(
        vad=_ScriptedVad(speech), policy=POLICY_P_BASED, turn_detector=_FixedTurn(0.0)
    ).run(near)
    patient = TurnEndpointDecider(
        vad=_ScriptedVad(speech), policy=POLICY_P_BASED, turn_detector=_FixedTurn(1.0)
    ).run(near)

    assert eager.stopped and patient.stopped
    # P=0 → required = floor (1000ms); P=1 → required = ceil (3000ms).
    assert eager.required_silence_ms == pytest.approx(1000.0)
    assert patient.required_silence_ms == pytest.approx(3000.0)
    assert eager.stop_frame < patient.stop_frame
    assert eager.trailing_silence_ms < patient.trailing_silence_ms


def test_p_based_requires_a_turn_detector() -> None:
    with pytest.raises(ValueError, match="requires a turn_detector"):
        TurnEndpointDecider(vad=_ScriptedVad(set()), policy=POLICY_P_BASED)


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown endpoint policy"):
        TurnEndpointDecider(vad=_ScriptedVad(set()), policy="magic")


def test_turn_never_stops_without_speech() -> None:
    near = _zeros(30, VAD_FRAME_SAMPLES)
    decider = TurnEndpointDecider(vad=_ScriptedVad(set()), policy=POLICY_FIXED)
    result = decider.run(near)

    assert not result.stopped
    assert result.speech_ms == pytest.approx(0.0)
    assert result.stop_ms is None
    # sanity: onset debounce constant is wired in from production
    assert VAD_FRAME_MS == pytest.approx(32.0)
