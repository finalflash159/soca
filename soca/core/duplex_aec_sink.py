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

from soca.core.audio_out import PlaybackResult, _resample, _to_float32_mono

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

    def play(
        self,
        audio: np.ndarray,
        sample_rate: int,
        blocking: bool = True,
        interrupt_event: Event | None = None,
    ) -> PlaybackResult:
        t0 = time.perf_counter()
        arr = _to_float32_mono(audio)
        if arr.size == 0:
            return PlaybackResult(False, sample_rate, 0.0, 0.0, "empty_audio")

        far_all = _resample(arr, sample_rate, self.rate)  # TTS rate -> 16k
        self._ensure_stream()
        assert self._stream is not None

        interrupted = False
        for start in range(0, len(far_all), self.frame):
            if interrupt_event is not None and interrupt_event.is_set():
                interrupted = True
                break

            far = far_all[start : start + self.frame]
            if len(far) < self.frame:
                far = np.concatenate([far, np.zeros(self.frame - len(far), np.float32)])

            # Duplex lockstep: write far + read near share one clock -> aligned.
            self._stream.write(far.reshape(-1, 1))
            near, _ = self._stream.read(self.frame)
            near = near.reshape(-1)[: self.frame].astype(np.float32)

            clean = self._aec.process(near, far)        # remove SoCa's own voice
            self._window.append(np.asarray(clean, dtype=np.float32))
            prob = float(self._model(torch.from_numpy(clean), self.rate).item())
            is_speech = prob >= self.vad_threshold
            self._run_ms = self._run_ms + self.block_ms if is_speech else 0.0
            if _DEBUG and is_speech:
                print(
                    f"[duplex] prob={prob:4.2f} run={self._run_ms:5.0f}/{self.sustained_ms:.0f}ms",
                    file=sys.stderr,
                    flush=True,
                )
            if self._run_ms >= self.sustained_ms and interrupt_event is not None:
                self.captured = np.concatenate(list(self._window)).astype(np.float32, copy=False)
                if _DEBUG:
                    print(f"[duplex] -> INTERRUPT (prob={prob:.2f})", file=sys.stderr, flush=True)
                interrupt_event.set()
                interrupted = True
                break

        latency_ms = (time.perf_counter() - t0) * 1000
        duration_ms = len(far_all) / self.rate * 1000
        return PlaybackResult(True, self.rate, duration_ms, latency_ms, interrupted=interrupted)

    def stop(self) -> None:
        """Close the duplex stream so the recorder can reopen the mic next turn."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
