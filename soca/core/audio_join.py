from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

CrossfadeCurve = Literal["raised_cosine_equal_gain", "equal_power"]


def _mono_f32(audio: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(audio, dtype=np.float32).reshape(-1))


def _fade_windows(
    samples: int,
    curve: CrossfadeCurve,
) -> tuple[np.ndarray, np.ndarray]:
    if samples <= 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty
    if samples == 1:
        half = np.array([0.5], dtype=np.float32)
        return half, half

    phase = np.linspace(
        0.0,
        np.pi / 2.0,
        samples,
        endpoint=True,
        dtype=np.float32,
    )
    if curve == "raised_cosine_equal_gain":
        fade_in = np.sin(phase) ** 2
        return 1.0 - fade_in, fade_in
    if curve == "equal_power":
        return np.cos(phase), np.sin(phase)
    raise ValueError(f"Unsupported cross-fade curve: {curve}")


def _fade_sample_count(
    audio_length: int,
    *,
    sample_rate: int,
    fade_ms: float,
) -> int:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if fade_ms < 0:
        raise ValueError("fade_ms must not be negative")
    return min(round(sample_rate * fade_ms / 1000.0), audio_length)


def crossfade_pcm(
    left: np.ndarray,
    right: np.ndarray,
    *,
    sample_rate: int,
    fade_ms: float = 12.0,
    curve: CrossfadeCurve = "raised_cosine_equal_gain",
) -> np.ndarray:
    """Join two already-resampled PCM buffers without a hard sample jump."""
    a, b = _mono_f32(left), _mono_f32(right)
    requested = _fade_sample_count(
        max(len(a), len(b)),
        sample_rate=sample_rate,
        fade_ms=fade_ms,
    )
    n = min(requested, len(a), len(b))
    if n == 0:
        return np.ascontiguousarray(np.concatenate((a, b)), dtype=np.float32)

    fade_out, fade_in = _fade_windows(n, curve)
    mixed = a[-n:] * fade_out + b[:n] * fade_in
    return np.ascontiguousarray(
        np.concatenate((a[:-n], mixed, b[n:])),
        dtype=np.float32,
    )


def fade_edge_pcm(
    audio: np.ndarray,
    *,
    sample_rate: int,
    fade_ms: float = 4.0,
    edge: Literal["in", "out"],
) -> np.ndarray:
    """Fade one edge for the non-overlapping late-chunk fallback."""
    result = _mono_f32(audio).copy()
    n = _fade_sample_count(
        len(result),
        sample_rate=sample_rate,
        fade_ms=fade_ms,
    )
    if n == 0:
        return result
    fade_out, fade_in = _fade_windows(n, "raised_cosine_equal_gain")
    if edge == "in":
        result[:n] *= fade_in
    elif edge == "out":
        result[-n:] *= fade_out
    else:
        raise ValueError(f"Unsupported fade edge: {edge}")
    return np.ascontiguousarray(result, dtype=np.float32)


@dataclass
class TailHoldingCrossfader:
    sample_rate: int
    fade_ms: float = 12.0
    curve: CrossfadeCurve = "raised_cosine_equal_gain"
    _tail: np.ndarray | None = None
    last_overlap_samples: int = 0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not 0.0 <= self.fade_ms <= 20.0:
            raise ValueError("fade_ms must be between 0 and 20")

    @property
    def fade_samples(self) -> int:
        return round(self.sample_rate * self.fade_ms / 1000.0)

    def _hold_size(self, audio: np.ndarray) -> int:
        requested = self.fade_samples
        if requested == 0 or len(audio) < 4 * requested:
            return 0
        return requested

    def push(self, audio: np.ndarray, *, fade_in: bool = False) -> np.ndarray:
        current = _mono_f32(audio)
        if fade_in:
            current = fade_edge_pcm(
                current,
                sample_rate=self.sample_rate,
                edge="in",
            )

        if self._tail is None:
            hold = self._hold_size(current)
            self._tail = (
                current[-hold:].copy()
                if hold
                else np.empty(0, dtype=np.float32)
            )
            self.last_overlap_samples = 0
            return np.ascontiguousarray(
                current[:-hold] if hold else current,
                dtype=np.float32,
            )

        previous_tail = self._tail
        overlap = min(len(previous_tail), len(current), self.fade_samples)
        self.last_overlap_samples = overlap
        if overlap:
            fade_out, next_fade_in = _fade_windows(overlap, self.curve)
            mixed = (
                previous_tail[-overlap:] * fade_out
                + current[:overlap] * next_fade_in
            )
            prefix = previous_tail[:-overlap]
        else:
            prefix = previous_tail
            mixed = np.empty(0, dtype=np.float32)

        hold = self._hold_size(current)
        body_end = max(overlap, len(current) - hold)
        self._tail = (
            current[body_end:].copy()
            if hold
            else np.empty(0, dtype=np.float32)
        )
        return np.ascontiguousarray(
            np.concatenate((prefix, mixed, current[overlap:body_end])),
            dtype=np.float32,
        )

    def finish(self, *, fade_out: bool = False) -> np.ndarray:
        tail = (
            self._tail
            if self._tail is not None
            else np.empty(0, dtype=np.float32)
        )
        self._tail = None
        self.last_overlap_samples = 0
        if fade_out and tail.size:
            return fade_edge_pcm(
                tail,
                sample_rate=self.sample_rate,
                edge="out",
            )
        return np.ascontiguousarray(tail, dtype=np.float32)

    def reset(self) -> None:
        self._tail = None
        self.last_overlap_samples = 0
