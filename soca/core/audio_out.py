from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock
from typing import Protocol

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly

INTERRUPT_FRAME_MS = 30


@dataclass(frozen=True)
class PlaybackResult:
    played: bool
    sample_rate: int
    audio_duration_ms: float
    latency_ms: float
    reason: str = ""
    interrupted: bool = False


class AudioSink(Protocol):
    def play(self, audio: np.ndarray, sample_rate: int, blocking: bool = True, interrupt_event: Event | None = None) -> PlaybackResult:
        ...

    def stop(self) -> None:
        ...


class AudioPlaybackSession(Protocol):
    sample_rate: int
    output_underflow_count: int

    def write(
        self,
        audio: np.ndarray,
        *,
        interrupt_event: Event | None = None,
    ) -> PlaybackResult:
        raise NotImplementedError

    def finish(
        self,
        *,
        interrupt_event: Event | None = None,
    ) -> PlaybackResult:
        raise NotImplementedError


class StreamingAudioSink(Protocol):
    def begin_playback(self, sample_rate: int) -> AudioPlaybackSession:
        raise NotImplementedError


def _to_float32_mono(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio)

    if arr.ndim == 0:
        return np.array([], dtype=np.float32)

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        max_abs = float(max(abs(info.min), info.max))
        arr = arr.astype(np.float32) / max_abs
    else:
        arr = arr.astype(np.float32, copy=False)

    arr = np.squeeze(arr)

    if arr.ndim == 2:
        arr = arr.mean(axis=1)

    return np.ascontiguousarray(arr, dtype=np.float32)


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio

    divisor = math.gcd(source_rate, target_rate)
    up = target_rate // divisor
    down = source_rate // divisor
    return resample_poly(audio, up, down).astype(np.float32, copy=False)


class NullAudioPlayer:
    def play(self, audio: np.ndarray, sample_rate: int, blocking: bool = True, interrupt_event: Event | None = None) -> PlaybackResult:
        arr = _to_float32_mono(audio)
        duration_ms = len(arr) / sample_rate * 1000 if sample_rate > 0 else 0.0
        return PlaybackResult(
            played=False,
            sample_rate=sample_rate,
            audio_duration_ms=duration_ms,
            latency_ms=0.0,
            reason="null_audio_player",
            interrupted=False,
        )

    def stop(self) -> None:
        return None


class _SoundDevicePlaybackSession:
    def __init__(
        self,
        sample_rate: int,
        *,
        block_ms: int = INTERRUPT_FRAME_MS,
        on_closed: Callable[[], None],
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.sample_rate = sample_rate
        self.output_underflow_count = 0
        self._block_samples = max(1, round(sample_rate * block_ms / 1000.0))
        self._on_closed = on_closed
        self._closed = False
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self._block_samples,
        )
        self._stream.start()

    def write(
        self,
        audio: np.ndarray,
        *,
        interrupt_event: Event | None = None,
    ) -> PlaybackResult:
        started = time.perf_counter()
        arr = _to_float32_mono(audio)
        if arr.size == 0:
            return PlaybackResult(
                False,
                self.sample_rate,
                0.0,
                0.0,
                "empty_audio",
            )
        if self._closed:
            raise RuntimeError("Audio playback session is already closed")

        written = 0
        interrupted = False
        for start in range(0, len(arr), self._block_samples):
            if interrupt_event is not None and interrupt_event.is_set():
                interrupted = True
                break
            block = arr[start : start + self._block_samples].reshape(-1, 1)
            underflowed = bool(self._stream.write(block))
            self.output_underflow_count += int(underflowed)
            written += len(block)

        latency_ms = (time.perf_counter() - started) * 1000.0
        return PlaybackResult(
            played=written > 0,
            sample_rate=self.sample_rate,
            audio_duration_ms=written / self.sample_rate * 1000.0,
            latency_ms=latency_ms,
            reason=("output_underflow" if self.output_underflow_count else ""),
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
        interrupted = interrupt_event is not None and interrupt_event.is_set()
        try:
            if interrupted:
                self._stream.abort()
            else:
                self._stream.stop()
        finally:
            self._stream.close()
            self._closed = True
            self._on_closed()
        return PlaybackResult(
            False,
            self.sample_rate,
            0.0,
            (time.perf_counter() - started) * 1000.0,
            interrupted=interrupted,
        )

    def abort(self) -> None:
        if self._closed:
            return
        try:
            self._stream.abort()
        finally:
            self._stream.close()
            self._closed = True
            self._on_closed()


class SoundDevicePlayer:
    def __init__(self, output_sample_rate: int | None = None) -> None:
        self.output_sample_rate = output_sample_rate
        self._session_lock = Lock()
        self._active_session: _SoundDevicePlaybackSession | None = None

    def begin_playback(self, sample_rate: int) -> AudioPlaybackSession:
        target_rate = self.output_sample_rate or sample_rate
        with self._session_lock:
            if self._active_session is not None:
                raise RuntimeError("A sounddevice playback session is already active")

            def clear_active() -> None:
                with self._session_lock:
                    self._active_session = None

            session = _SoundDevicePlaybackSession(
                target_rate,
                on_closed=clear_active,
            )
            self._active_session = session
            return session

    def _play_interruptible(self, arr, target_rate, interrupt_event) -> bool:
        frame = max(1, int(INTERRUPT_FRAME_MS * target_rate / 1000))

        with sd.OutputStream(samplerate=target_rate, channels=1, dtype="float32") as stream:
            for start in range(0, len(arr), frame):
                if interrupt_event.is_set():
                    stream.abort() # remove the rest of the buffer
                    return True
                stream.write(arr[start: start + frame])
        return False

    def play(self, audio: np.ndarray, sample_rate: int, blocking: bool = True, interrupt_event: Event | None = None) -> PlaybackResult:
        t0 = time.perf_counter()

        arr = _to_float32_mono(audio)
        if arr.size == 0:
            return PlaybackResult(False, sample_rate, 0.0, 0.0, "empty_audio", interrupted=False)

        target_rate = self.output_sample_rate or sample_rate
        arr = _resample(arr, sample_rate, target_rate)

        if interrupt_event is not None:
            interrupted = self._play_interruptible(arr, target_rate, interrupt_event)
            latency_ms = (time.perf_counter() - t0) * 1000
            duration_ms = len(arr) / target_rate * 1000
            return PlaybackResult(True, target_rate, duration_ms, latency_ms, interrupted=interrupted)

        sd.play(arr, samplerate=target_rate, blocking=blocking)

        latency_ms = (time.perf_counter() - t0) * 1000
        duration_ms = len(arr) / target_rate * 1000

        return PlaybackResult(True, target_rate, duration_ms, latency_ms)

    def stop(self) -> None:
        with self._session_lock:
            session = self._active_session
        if session is not None:
            session.abort()
        else:
            sd.stop()


class WavFileSink:
    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)

    def play(self, audio: np.ndarray, sample_rate: int, blocking: bool = True, interrupt_event: Event | None = None) -> PlaybackResult:
        t0 = time.perf_counter()

        arr = _to_float32_mono(audio)
        if arr.size == 0:
            return PlaybackResult(False, sample_rate, 0.0, 0.0, "empty_audio")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(self.output_path, arr, sample_rate)

        latency_ms = (time.perf_counter() - t0) * 1000
        duration_ms = len(arr) / sample_rate * 1000

        return PlaybackResult(True, sample_rate, duration_ms, latency_ms, f"saved:{self.output_path}")

    def stop(self) -> None:
        return None
