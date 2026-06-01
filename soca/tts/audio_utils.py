from __future__ import annotations

import io
import time
from typing import Any

import numpy as np
import soundfile as sf

from .base import TTSResult


def to_float32_mono(audio: Any) -> np.ndarray:
    arr = np.asarray(audio)
    if arr.ndim == 0:
        return np.array([], dtype=np.float32)

    arr = np.squeeze(arr)
    if arr.ndim == 2:
        if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[1]:
            arr = arr.mean(axis=0)
        else:
            arr = arr.mean(axis=1)

    arr = arr.astype(np.float32, copy=False)
    if arr.size:
        peak = float(np.max(np.abs(arr)))
        if peak > 1.5:
            arr = arr / peak
    return np.ascontiguousarray(arr, dtype=np.float32)


def wav_bytes_to_float32_mono(payload: bytes) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
    return to_float32_mono(audio), int(sample_rate)


def empty_result(text: str, voice: str, engine: str, sample_rate: int = 24000) -> TTSResult:
    return TTSResult(
        text=text,
        audio=np.array([], dtype=np.float32),
        sample_rate=sample_rate,
        latency_ms=0.0,
        audio_duration_ms=0.0,
        rtf=0.0,
        voice=voice,
        engine=engine,
    )


def build_result(
    *,
    text: str,
    audio: Any,
    sample_rate: int,
    started_at: float,
    voice: str,
    engine: str,
) -> TTSResult:
    audio_np = to_float32_mono(audio)
    latency_ms = (time.perf_counter() - started_at) * 1000
    duration_ms = (len(audio_np) / sample_rate) * 1000 if sample_rate > 0 else 0.0
    return TTSResult(
        text=text,
        audio=audio_np,
        sample_rate=sample_rate,
        latency_ms=latency_ms,
        audio_duration_ms=duration_ms,
        rtf=latency_ms / duration_ms if duration_ms > 0 else 0.0,
        voice=voice,
        engine=engine,
    )

