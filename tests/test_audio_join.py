from __future__ import annotations

import numpy as np
import pytest

from soca.core.audio_join import (
    TailHoldingCrossfader,
    crossfade_pcm,
    fade_edge_pcm,
)


def test_crossfade_is_float32_contiguous_and_has_expected_length() -> None:
    left = np.ones(480, dtype=np.float32)
    right = -np.ones(480, dtype=np.float32)

    joined = crossfade_pcm(left, right, sample_rate=24_000, fade_ms=10)

    assert joined.dtype == np.float32
    assert joined.flags.c_contiguous
    assert len(joined) == len(left) + len(right) - 240


def test_crossfade_removes_hard_boundary_jump() -> None:
    left = np.full(480, 0.8, dtype=np.float32)
    right = np.full(480, -0.8, dtype=np.float32)

    joined = crossfade_pcm(left, right, sample_rate=24_000, fade_ms=10)

    assert abs(float(left[-1] - right[0])) == pytest.approx(1.6)
    assert float(np.abs(np.diff(joined)).max()) < 0.25


def test_equal_gain_crossfade_does_not_raise_identical_signal_peak() -> None:
    left = np.full(480, 0.75, dtype=np.float32)
    right = np.full(480, 0.75, dtype=np.float32)

    joined = crossfade_pcm(left, right, sample_rate=24_000, fade_ms=10)

    assert float(np.abs(joined).max()) == pytest.approx(0.75, abs=1e-6)


def test_zero_fade_is_plain_concatenation() -> None:
    left = np.array([0.1, 0.2], dtype=np.float32)
    right = np.array([0.3], dtype=np.float32)

    joined = crossfade_pcm(left, right, sample_rate=24_000, fade_ms=0)

    np.testing.assert_array_equal(joined, np.array([0.1, 0.2, 0.3], np.float32))


def test_tail_holding_matches_offline_reference() -> None:
    first = np.linspace(-0.4, 0.4, 1_200, dtype=np.float32)
    second = np.linspace(0.4, -0.4, 1_200, dtype=np.float32)
    joiner = TailHoldingCrossfader(sample_rate=24_000, fade_ms=12)

    streamed = np.concatenate(
        (joiner.push(first), joiner.push(second), joiner.finish())
    )
    expected = crossfade_pcm(first, second, sample_rate=24_000, fade_ms=12)

    np.testing.assert_allclose(streamed, expected, atol=1e-6)


def test_short_chunk_is_not_held_for_ttfa() -> None:
    short = np.ones(500, dtype=np.float32)
    joiner = TailHoldingCrossfader(sample_rate=24_000, fade_ms=12)

    emitted = joiner.push(short)

    np.testing.assert_array_equal(emitted, short)
    assert joiner.finish().size == 0


def test_non_overlapping_fallback_fades_both_edges() -> None:
    audio = np.ones(480, dtype=np.float32)

    faded_out = fade_edge_pcm(
        audio,
        sample_rate=24_000,
        fade_ms=4,
        edge="out",
    )
    faded_in = fade_edge_pcm(
        audio,
        sample_rate=24_000,
        fade_ms=4,
        edge="in",
    )

    assert faded_out[-1] == pytest.approx(0.0, abs=1e-6)
    assert faded_in[0] == pytest.approx(0.0, abs=1e-6)
