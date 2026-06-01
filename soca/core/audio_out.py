from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly


@dataclass(frozen=True)
class PlaybackResult:
    played: bool
    sample_rate: int
    audio_duration_ms: float
    latency_ms: float
    reason: str = ""


class AudioSink(Protocol):
    def play(self, audio: np.ndarray, sample_rate: int, blocking: bool = True) -> PlaybackResult:
        ...

    def stop(self) -> None:
        ...


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
    def play(self, audio: np.ndarray, sample_rate: int, blocking: bool = True) -> PlaybackResult:
        arr = _to_float32_mono(audio)
        duration_ms = len(arr) / sample_rate * 1000 if sample_rate > 0 else 0.0
        return PlaybackResult(
            played=False,
            sample_rate=sample_rate,
            audio_duration_ms=duration_ms,
            latency_ms=0.0,
            reason="null_audio_player",
        )

    def stop(self) -> None:
        return None


class SoundDevicePlayer:
    def __init__(self, output_sample_rate: int | None = None) -> None:
        self.output_sample_rate = output_sample_rate

    def play(self, audio: np.ndarray, sample_rate: int, blocking: bool = True) -> PlaybackResult:
        t0 = time.perf_counter()

        arr = _to_float32_mono(audio)
        if arr.size == 0:
            return PlaybackResult(False, sample_rate, 0.0, 0.0, "empty_audio")

        target_rate = self.output_sample_rate or sample_rate
        arr = _resample(arr, sample_rate, target_rate)

        sd.play(arr, samplerate=target_rate, blocking=blocking)

        latency_ms = (time.perf_counter() - t0) * 1000
        duration_ms = len(arr) / target_rate * 1000

        return PlaybackResult(True, target_rate, duration_ms, latency_ms)

    def stop(self) -> None:
        sd.stop()


class WavFileSink:
    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)

    def play(self, audio: np.ndarray, sample_rate: int, blocking: bool = True) -> PlaybackResult:
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
