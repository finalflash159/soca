"""Offline, frame-stepped replay of the conversational decision logic (P3.1 Pha A).

The live path couples two decisions to sounddevice streams — barge-in in
``DuplexAecSink._process_far_frames`` (mic ``read`` + speaker ``write``) and
turn-taking in ``record_until_silence`` (mic ``read``). Neither can be measured
reproducibly while it owns hardware.

This module lifts *only the decision arithmetic* out of those loops and drives it
from supplied ``(far, near)`` buffers instead of a live device. The one rule that
makes it a benchmark rather than a re-implementation: **time is frame index, not
wall clock** — frame ``i`` covers audio ``[i·block_ms, (i+1)·block_ms)`` — so a run
is fully determined by its inputs and independent of the machine it runs on.

Both the echo canceller and the VAD are *injected* (``EchoCanceller`` / ``SpeechProb``
protocols), so the deciders are unit-testable with synthetic frames — no
``pywebrtc_audio``, no Silero, no ONNX in CI. Pha B feeds them real AEC-Challenge /
RIR audio through the same seams.

Faithfulness anchors (keep these in lockstep with production):
  - barge-in: ``soca/core/duplex_aec_sink.py::_process_far_frames``
  - turn-taking: ``soca/core/endpoint.py::_decide_required_silence`` +
    ``soca/core/turn_taking.py::required_silence_from_p``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from soca.core.endpoint import EndpointConfig
from soca.core.turn_taking import (
    SPEECH_CONFIRM_FRAMES,
    VAD_FRAME_MS,
    VAD_FRAME_SAMPLES,
    required_silence_from_p,
)

_SAMPLE_RATE = 16000
_BARGE_BLOCK_MS = 32  # DuplexAecSink default: 512 samples @16k


class EchoCanceller(Protocol):
    """WebRTC-style AEC: cancel ``far`` (speaker reference) out of ``near`` (mic)."""

    def process(self, near: np.ndarray, far: np.ndarray) -> np.ndarray: ...


class SpeechProb(Protocol):
    """A VAD reduced to one number: P(speech) for a single frame."""

    def __call__(self, frame: np.ndarray, sample_rate: int) -> float: ...


class TurnDetector(Protocol):
    """Smart-Turn surface: P(user is still speaking) for a voiced window."""

    def p_still_speaking(self, audio_window: np.ndarray) -> float: ...


# --------------------------------------------------------------------------- #
# Tier 1 — barge-in (acoustic front-end: AEC + VAD + sustained run)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BargeInResult:
    """Outcome of replaying one ``(far, near)`` pair through the barge-in gate."""

    interrupted: bool
    interrupt_frame: int | None  # 0-based index of the frame that fired
    interrupt_ms: float | None  # audio-time the interrupt was detected = (frame+1)·block_ms
    frames_processed: int
    speech_frames: int  # frames whose post-AEC prob cleared the threshold
    max_run_ms: float  # longest sustained speech run seen (diagnostic)


class BargeInDecider:
    """Frame-stepped replay of ``DuplexAecSink`` barge-in detection.

    Mirrors ``_process_far_frames`` exactly — per frame: ``clean = aec.process(near,
    far)`` → ``prob = vad(clean)`` → accumulate a speech run → fire once the run
    reaches ``sustained_ms`` — but reads ``near`` from a buffer, not a mic. Pure over
    its inputs: same ``(far, near)`` in, same result out.
    """

    def __init__(
        self,
        *,
        aec: EchoCanceller,
        vad: SpeechProb,
        sample_rate: int = _SAMPLE_RATE,
        block_ms: int = _BARGE_BLOCK_MS,
        sustained_ms: float = 400.0,
        vad_threshold: float = 0.7,
    ) -> None:
        self.rate = sample_rate
        self.block_ms = block_ms
        self.frame = int(sample_rate * block_ms / 1000)
        self.sustained_ms = float(sustained_ms)
        self.vad_threshold = float(vad_threshold)
        self._aec = aec
        self._vad = vad

    def run(self, far: np.ndarray, near: np.ndarray) -> BargeInResult:
        """Replay a whole turn's worth of paired frames, stopping at the first fire."""
        far_arr = np.asarray(far, dtype=np.float32).reshape(-1)
        near_arr = np.asarray(near, dtype=np.float32).reshape(-1)
        if len(far_arr) != len(near_arr):
            raise ValueError("far and near must be the same length for aligned replay")

        n_frames = len(far_arr) // self.frame
        run_ms = 0.0
        max_run_ms = 0.0
        speech_frames = 0

        for i in range(n_frames):
            sl = slice(i * self.frame, (i + 1) * self.frame)
            far_frame = np.ascontiguousarray(far_arr[sl], dtype=np.float32)
            near_frame = np.ascontiguousarray(near_arr[sl], dtype=np.float32)

            clean = np.asarray(self._aec.process(near_frame, far_frame), dtype=np.float32)
            is_speech = float(self._vad(clean, self.rate)) >= self.vad_threshold
            run_ms = run_ms + self.block_ms if is_speech else 0.0
            if is_speech:
                speech_frames += 1
            max_run_ms = max(max_run_ms, run_ms)

            if run_ms >= self.sustained_ms:
                return BargeInResult(
                    interrupted=True,
                    interrupt_frame=i,
                    interrupt_ms=(i + 1) * self.block_ms,
                    frames_processed=i + 1,
                    speech_frames=speech_frames,
                    max_run_ms=max_run_ms,
                )

        return BargeInResult(
            interrupted=False,
            interrupt_frame=None,
            interrupt_ms=None,
            frames_processed=n_frames,
            speech_frames=speech_frames,
            max_run_ms=max_run_ms,
        )


# --------------------------------------------------------------------------- #
# Tier 2 — turn-taking (endpoint: silence run vs a policy-chosen threshold)
# --------------------------------------------------------------------------- #

# Endpoint policies. Each maps the current pause to a required trailing-silence (ms):
#   fixed   — constant endpoint_silence_ms (non-adaptive baseline)
#   p_based — floor + span·P(still-speaking); the production adaptive path
POLICY_FIXED = "fixed"
POLICY_P_BASED = "p_based"


@dataclass(frozen=True)
class TurnResult:
    """Outcome of replaying one user utterance through the endpoint gate."""

    stopped: bool
    stop_frame: int | None  # 0-based VAD-frame index the turn closed on
    stop_ms: float | None  # audio-time the turn closed = (frame+1)·VAD_FRAME_MS
    policy: str
    speech_ms: float  # confirmed speech heard before the close
    trailing_silence_ms: float  # silence run at the close (≈ the required threshold)
    required_silence_ms: float | None  # the threshold that was met (diagnostic)


class TurnEndpointDecider:
    """Frame-stepped replay of the adaptive endpoint decision.

    Reuses the production silence bookkeeping (Silero streaming frame =
    ``VAD_FRAME_SAMPLES`` with ``SPEECH_CONFIRM_FRAMES`` onset debounce, mirroring
    ``IncrementalVadTracker``) and the production threshold formula
    (``required_silence_from_p``), so the only thing swapped out is the live mic.

    The endpoint model gate mirrors ``_decide_required_silence``: while the pause is
    shorter than ``floor_silence_ms`` the decider stays patient (``ceil_silence_ms``);
    past the floor it asks the turn detector and uses ``floor + span·P``. ``fixed``
    ignores the model and waits a constant ``endpoint_silence_ms``.

    VAD and turn detector are injected, so tests need neither Silero nor Smart-Turn.
    """

    def __init__(
        self,
        *,
        vad: SpeechProb,
        policy: str = POLICY_P_BASED,
        turn_detector: TurnDetector | None = None,
        config: EndpointConfig | None = None,
        vad_threshold: float = 0.5,
    ) -> None:
        if policy not in (POLICY_FIXED, POLICY_P_BASED):
            raise ValueError(f"unknown endpoint policy: {policy!r}")
        if policy == POLICY_P_BASED and turn_detector is None:
            raise ValueError("p_based policy requires a turn_detector")
        self._vad = vad
        self.policy = policy
        self._turn = turn_detector
        self.config = config or EndpointConfig()
        self.vad_threshold = float(vad_threshold)
        self.frame = VAD_FRAME_SAMPLES

    def _required_silence_ms(self, silence_ms: float, voiced: np.ndarray) -> float:
        """Mirror ``_decide_required_silence`` for the active policy."""
        cfg = self.config
        if self.policy == POLICY_FIXED:
            return float(cfg.endpoint_silence_ms)
        # p_based: patient until the floor, then modulate by the turn model.
        if silence_ms < cfg.floor_silence_ms:
            return float(cfg.ceil_silence_ms)
        if voiced.size == 0:
            return float(cfg.ceil_silence_ms)
        assert self._turn is not None
        p = self._turn.p_still_speaking(voiced)
        return required_silence_from_p(p, cfg)

    def run(self, near: np.ndarray) -> TurnResult:
        """Replay a user utterance frame by frame; stop when silence clears the bar."""
        near_arr = np.asarray(near, dtype=np.float32).reshape(-1)
        n_frames = len(near_arr) // self.frame

        streak = 0
        speech_ms = 0.0
        silence_run_ms = 0.0
        has_speech = False
        max_voiced_samples = int(8.0 * self.config.sample_rate)  # Smart-Turn 8s cap

        for i in range(n_frames):
            frame = near_arr[i * self.frame : (i + 1) * self.frame]
            is_speech = float(self._vad(frame, self.config.sample_rate)) >= self.vad_threshold

            if is_speech:
                streak += 1
                if streak == SPEECH_CONFIRM_FRAMES:  # onset: count the confirm window
                    has_speech = True
                    speech_ms += VAD_FRAME_MS * SPEECH_CONFIRM_FRAMES
                    silence_run_ms = 0.0
                elif streak > SPEECH_CONFIRM_FRAMES:
                    speech_ms += VAD_FRAME_MS
                    silence_run_ms = 0.0
            else:
                streak = 0
                if has_speech:
                    silence_run_ms += VAD_FRAME_MS

            if not has_speech:
                continue

            # Voiced window = speech heard before the current pause, capped at 8s.
            voiced_end = (i + 1) * self.frame - int(silence_run_ms / 1000 * self.config.sample_rate)
            voiced_start = max(0, voiced_end - max_voiced_samples)
            voiced = near_arr[voiced_start:voiced_end] if voiced_end > 0 else near_arr[:0]

            required = self._required_silence_ms(silence_run_ms, voiced)
            if silence_run_ms >= required:
                return TurnResult(
                    stopped=True,
                    stop_frame=i,
                    stop_ms=(i + 1) * VAD_FRAME_MS,
                    policy=self.policy,
                    speech_ms=speech_ms,
                    trailing_silence_ms=silence_run_ms,
                    required_silence_ms=required,
                )

        return TurnResult(
            stopped=False,
            stop_frame=None,
            stop_ms=None,
            policy=self.policy,
            speech_ms=speech_ms,
            trailing_silence_ms=silence_run_ms,
            required_silence_ms=None,
        )
