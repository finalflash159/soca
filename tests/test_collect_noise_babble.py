"""Tests for speech-like babble synthesis (P1.1 Pha B).

Pure array functions only — no HF datasets, no FLEURS files on disk.
"""

from __future__ import annotations

import numpy as np
import pytest

from local.collect_noise import (
    _extract_segment,
    mix_babble,
    normalize_rms,
    rms,
)


def test_normalize_rms_hits_target_and_guards_peak() -> None:
    rng = np.random.default_rng(0)
    audio = rng.normal(0.0, 0.3, 8000).astype(np.float32)

    out = normalize_rms(audio, target_rms=0.05)

    assert rms(out) == pytest.approx(0.05, rel=1e-3)
    assert np.max(np.abs(out)) <= 0.99 + 1e-6
    assert out.dtype == np.float32


def test_normalize_rms_leaves_silence_silent() -> None:
    out = normalize_rms(np.zeros(1000, dtype=np.float32), target_rms=0.05)

    assert not np.any(np.isnan(out))
    assert np.all(out == 0.0)


def test_mix_babble_length_dtype_and_loudness() -> None:
    rng = np.random.default_rng(1)
    segments = [rng.normal(0.0, 0.2, 4000).astype(np.float32) for _ in range(4)]

    out = mix_babble(segments, rng, target_rms=0.05, reverse_prob=0.5)

    assert len(out) == 4000
    assert out.dtype == np.float32
    assert rms(out) == pytest.approx(0.05, rel=1e-3)


def test_mix_babble_reverses_single_voice_when_prob_one() -> None:
    seg = np.linspace(-0.5, 0.5, 2000, dtype=np.float32)

    out = mix_babble([seg], np.random.default_rng(3), target_rms=0.05, reverse_prob=1.0)

    # Gain cancels under RMS normalisation, so the result is just the normalised reverse.
    assert np.allclose(out, normalize_rms(seg[::-1], 0.05), atol=1e-6)


def test_mix_babble_is_deterministic_under_fixed_seed() -> None:
    segments = [np.linspace(-0.3, 0.3, 3000, dtype=np.float32) + i for i in range(3)]

    a = mix_babble(segments, np.random.default_rng(7), target_rms=0.05, reverse_prob=0.5)
    b = mix_babble(segments, np.random.default_rng(7), target_rms=0.05, reverse_prob=0.5)

    assert np.array_equal(a, b)


def test_mix_babble_rejects_bad_input() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="at least one segment"):
        mix_babble([], rng, target_rms=0.05, reverse_prob=0.5)
    with pytest.raises(ValueError, match="one length"):
        mix_babble(
            [np.zeros(10, dtype=np.float32), np.zeros(11, dtype=np.float32)],
            rng,
            target_rms=0.05,
            reverse_prob=0.5,
        )


def test_extract_segment_tiles_short_source() -> None:
    short = np.arange(100, dtype=np.float32)

    seg = _extract_segment(short, n_samples=250, rng=np.random.default_rng(0))

    assert len(seg) == 250


def test_extract_segment_is_a_contiguous_window() -> None:
    audio = np.arange(1000, dtype=np.float32)

    seg = _extract_segment(audio, n_samples=200, rng=np.random.default_rng(0))

    assert len(seg) == 200
    # Values are consecutive integers → a real slice, not a shuffle.
    assert np.array_equal(seg, np.arange(seg[0], seg[0] + 200, dtype=np.float32))
