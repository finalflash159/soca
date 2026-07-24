from __future__ import annotations

import os
import sys
import time
from collections import deque
from threading import Event

import numpy as np
import sounddevice as sd
import torch
from pywebrtc_audio import AudioProcessor
from silero_vad import load_silero_vad

from soca.core.audio_out import (
    AudioPlaybackSession,
    PlaybackResult,
    _resample,
    _to_float32_mono,
)

_SAMPLE_RATE = 16000
_FRAME = 512  # 32 ms @16k == Silero VAD frame
_BLOCK_MS = 32
_CAPTURE_WINDOW_MS = 1200

# Opt-in debug: SOCA_BARGE_DEBUG=1 prints the (post-AEC) speech probability so you
# can confirm echo is cancelled (prob stays low while SoCa speaks).
_DEBUG = os.environ.get("SOCA_BARGE_DEBUG", "") not in ("", "0")


class DuplexAecSink:
    """Path B sink: play TTS through a PERSISTENT duplex stream while capturing the
    mic, cancelling echo (WebRTC AEC3), and detecting barge-in inline — all on a
    SINGLE clock so AEC stays converged (what Path A's two-stream design could not do).

    Replaces ``SoundDevicePlayer`` + the separate ``BargeInListener`` thread. The
    duplex stream opens on the first ``play`` of a turn and stays open across the
    turn's chunks; ``stop()`` closes it so the recorder can reclaim the mic.

    Implements the ``AudioSink`` protocol (``play``/``stop``). On sustained speech it
    sets ``interrupt_event`` and stores the (echo-cancelled) interrupting audio in
    ``captured`` for the caller to prepend to the next recording.
    """

    def __init__(
        self,
        *,
        sample_rate: int = _SAMPLE_RATE,
        block_ms: int = _BLOCK_MS,
        sustained_ms: float = 400,
        vad_threshold: float = 0.7,
        stream_delay_ms: int = 40,
    ) -> None:
        self.rate = sample_rate
        self.frame = int(sample_rate * block_ms / 1000)
        self.block_ms = block_ms
        # Env overrides for live tuning (same knobs as the barge-in listener).
        self.sustained_ms = float(os.environ.get("SOCA_BARGE_SUSTAINED_MS", sustained_ms))
        self.vad_threshold = float(os.environ.get("SOCA_BARGE_THRESHOLD", vad_threshold))
        self._model = load_silero_vad()
        self._aec = AudioProcessor(
            sample_rate=sample_rate,
            num_channels=1,
            echo_cancellation=True,
            noise_suppression=True,
            auto_gain_control=False,
            stream_delay_ms=int(os.environ.get("SOCA_AEC_DELAY_MS", stream_delay_ms)),
        )
        self._stream: sd.Stream | None = None
        self._run_ms = 0.0
        self._window: deque[np.ndarray] = deque(maxlen=max(1, int(_CAPTURE_WINDOW_MS / block_ms)))
        # Echo-cancelled audio that triggered the last interrupt; caller prepends it
        # to the next recording so the user's first words survive.
        self.captured: np.ndarray | None = None
        self._playback_session: _DuplexPlaybackSession | None = None

    def _ensure_stream(self) -> None:
        """Open the duplex stream + reset per-turn state on the first chunk."""
        if self._stream is not None:
            return
        self._model.reset_states()
        self._run_ms = 0.0
        self._window.clear()
        self.captured = None
        self._stream = sd.Stream(
            samplerate=self.rate, blocksize=self.frame, channels=1, dtype="float32"
        )
        self._stream.start()

    def begin_playback(self, sample_rate: int) -> AudioPlaybackSession:
        """Open the persistent duplex session for one turn of joined PCM."""
        del sample_rate
        if self._playback_session is not None:
            raise RuntimeError("A duplex playback session is already active")
        self._ensure_stream()
        session = _DuplexPlaybackSession(self)
        self._playback_session = session
        return session

    def _release_playback_session(self, session: _DuplexPlaybackSession) -> None:
        if self._playback_session is session:
            self._playback_session = None

    def _process_far_frames(
        self,
        far_audio: np.ndarray,
        *,
        interrupt_event: Event | None,
    ) -> tuple[bool, int]:
        """Play + AEC + VAD complete duplex frames; no padding inside the loop."""
        if len(far_audio) % self.frame != 0:
            raise ValueError("far_audio must contain complete duplex frames")
        self._ensure_stream()
        assert self._stream is not None

        consumed = 0
        for start in range(0, len(far_audio), self.frame):
            if interrupt_event is not None and interrupt_event.is_set():
                return True, consumed

            far = np.ascontiguousarray(
                far_audio[start : start + self.frame],
                dtype=np.float32,
            )
            # Duplex lockstep: write far + read near share one clock -> aligned.
            self._stream.write(far.reshape(-1, 1))
            near, _ = self._stream.read(self.frame)
            near = near.reshape(-1)[: self.frame].astype(np.float32, copy=False)

            clean = np.asarray(self._aec.process(near, far), dtype=np.float32)
            self._window.append(clean)
            prob = float(self._model(torch.from_numpy(clean), self.rate).item())
            is_speech = prob >= self.vad_threshold
            self._run_ms = self._run_ms + self.block_ms if is_speech else 0.0
            consumed += self.frame
            if _DEBUG and is_speech:
                print(
                    f"[duplex] prob={prob:4.2f} "
                    f"run={self._run_ms:5.0f}/{self.sustained_ms:.0f}ms",
                    file=sys.stderr,
                    flush=True,
                )
            if self._run_ms >= self.sustained_ms and interrupt_event is not None:
                self.captured = np.concatenate(list(self._window)).astype(
                    np.float32,
                    copy=False,
                )
                if _DEBUG:
                    print(f"[duplex] -> INTERRUPT (prob={prob:.2f})", file=sys.stderr, flush=True)
                interrupt_event.set()
                return True, consumed
        return False, consumed

    def play(
        self,
        audio: np.ndarray,
        sample_rate: int,
        blocking: bool = True,
        interrupt_event: Event | None = None,
    ) -> PlaybackResult:
        """One-shot fallback: resample, then run a full session for one buffer."""
        del blocking
        t0 = time.perf_counter()
        arr = _to_float32_mono(audio)
        if arr.size == 0:
            return PlaybackResult(False, sample_rate, 0.0, 0.0, "empty_audio")

        far = _resample(arr, sample_rate, self.rate)  # TTS rate -> 16k
        session = self.begin_playback(self.rate)
        written = session.write(far, interrupt_event=interrupt_event)
        finished = session.finish(interrupt_event=interrupt_event)
        return PlaybackResult(
            played=written.played or finished.played,
            sample_rate=self.rate,
            audio_duration_ms=len(far) / self.rate * 1000.0,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            reason=written.reason or finished.reason,
            interrupted=written.interrupted or finished.interrupted,
        )

    def stop(self) -> None:
        """Close the duplex stream so the recorder can reopen the mic next turn."""
        if self._playback_session is not None:
            self._playback_session.cancel()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None


class _DuplexPlaybackSession:
    """One turn of joined PCM played through a persistent duplex stream.

    Buffers the sub-frame remainder between ``write`` calls so two adjacent
    chunks never get a zero-pad wedged between them; only ``finish`` pads the
    final partial frame once, at end of turn.
    """

    def __init__(self, sink: DuplexAecSink) -> None:
        self.sample_rate = sink.rate
        self.output_underflow_count = 0
        self._sink = sink
        self._residual = np.empty(0, dtype=np.float32)
        self._closed = False

    def write(
        self,
        audio: np.ndarray,
        *,
        interrupt_event: Event | None = None,
    ) -> PlaybackResult:
        if self._closed:
            raise RuntimeError("Duplex playback session is already closed")
        started = time.perf_counter()
        current = _to_float32_mono(audio)
        combined = np.ascontiguousarray(
            np.concatenate((self._residual, current)),
            dtype=np.float32,
        )
        full_length = len(combined) // self._sink.frame * self._sink.frame
        complete = combined[:full_length]
        self._residual = combined[full_length:].copy()
        if complete.size:
            interrupted, consumed = self._sink._process_far_frames(
                complete,
                interrupt_event=interrupt_event,
            )
        else:
            interrupted, consumed = False, 0
        if interrupted:
            self._residual = np.empty(0, dtype=np.float32)
        return PlaybackResult(
            played=consumed > 0,
            sample_rate=self.sample_rate,
            audio_duration_ms=consumed / self.sample_rate * 1000.0,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            interrupted=interrupted,
        )

    def finish(
        self,
        *,
        interrupt_event: Event | None = None,
    ) -> PlaybackResult:
        started = time.perf_counter()
        if self._closed:
            return PlaybackResult(False, self.sample_rate, 0.0, 0.0, "already_closed")

        requested = len(self._residual)
        interrupted = interrupt_event is not None and interrupt_event.is_set()
        consumed = 0
        try:
            if requested and not interrupted:
                padded = np.pad(
                    self._residual,
                    (0, self._sink.frame - requested),
                ).astype(np.float32, copy=False)
                interrupted, consumed = self._sink._process_far_frames(
                    padded,
                    interrupt_event=interrupt_event,
                )
        finally:
            self._residual = np.empty(0, dtype=np.float32)
            self._closed = True
            self._sink._release_playback_session(self)
        return PlaybackResult(
            played=consumed > 0,
            sample_rate=self.sample_rate,
            audio_duration_ms=requested / self.sample_rate * 1000.0,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            interrupted=interrupted,
        )

    def cancel(self) -> None:
        if self._closed:
            return
        self._residual = np.empty(0, dtype=np.float32)
        self._closed = True
        self._sink._release_playback_session(self)
