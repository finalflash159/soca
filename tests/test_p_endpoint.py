from __future__ import annotations

import numpy as np
import pytest

from soca.core.turn_taking import (
    acoustic_filler_score,
    energy_trailing_score,
    estimate_p_still_speaking,
    required_silence_from_p,
)

SR = 16000


def _tone(freq: float, ms: float, amp=1.0) -> np.ndarray:
    n = int(SR * ms / 1000)
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class Cfg:
    floor_silence_ms = 500
    ceil_silence_ms = 3000


# ---------- energy_trailing_score ----------

def test_energy_constant_tone_scores_high():
    # loudness flat to the very edge -> abrupt stop -> "still going"
    assert energy_trailing_score(_tone(200, 500)) > 0.9


def test_energy_decay_to_silence_scores_low():
    # linear fade-out -> final lowering -> "done"
    x = _tone(200, 500)
    x *= np.linspace(1.0, 0.0, len(x)).astype(np.float32)
    assert energy_trailing_score(x) < 0.2


def test_energy_too_short_is_zero():
    assert energy_trailing_score(_tone(200, 5)) == 0.0


# ---------- acoustic_filler_score ----------

def test_filler_long_steady_tone_scores_high():
    # a held vowel: spectrum barely changes + long enough
    assert acoustic_filler_score(_tone(220, 500)) > 0.7


def test_filler_white_noise_scores_low():
    # spectrum churns frame-to-frame -> high flux -> not a held sound
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(int(SR * 0.5)).astype(np.float32)
    assert acoustic_filler_score(noise) < 0.2


def test_filler_short_tone_scores_low():
    # steady but too short to be a hesitation "ummm"
    assert acoustic_filler_score(_tone(220, 150)) < 0.2


def test_filler_chirp_scores_low():
    # sweeping spectrum -> formants "moving" -> articulated speech, not a filler
    n = int(SR * 0.5)
    t = np.arange(n) / SR
    chirp = np.sin(2 * np.pi * (150 + 1500 * t) * t).astype(np.float32)
    assert acoustic_filler_score(chirp) < 0.4


# ---------- required_silence_from_p ----------

def test_required_silence_maps_p_linearly():
    cfg = Cfg()
    assert required_silence_from_p(0.0, cfg) == 500
    assert required_silence_from_p(1.0, cfg) == 3000
    assert required_silence_from_p(0.5, cfg) == pytest.approx(1750)


def test_required_silence_clamps_out_of_range():
    cfg = Cfg()
    assert required_silence_from_p(-1.0, cfg) == 500
    assert required_silence_from_p(2.0, cfg) == 3000


# ---------- estimate_p_still_speaking ----------

def test_estimate_p_incomplete_text_raises_patience():
    # connective ending, no audio -> semantic cue alone
    assert estimate_p_still_speaking("tôi muốn nói với", None) == pytest.approx(0.8)


def test_estimate_p_complete_text_and_quiet_tail_is_low():
    # complete-looking text + tapered tail -> nothing says "still going"
    x = _tone(200, 500) * np.linspace(1.0, 0.0, int(SR * 0.5)).astype(np.float32)
    assert estimate_p_still_speaking("bật đèn lên", x) < 0.3


def test_estimate_p_held_vowel_tail_raises_patience_without_text():
    # transcript looks done, but the tail is a long held sound -> acoustic saves it
    p = estimate_p_still_speaking("cho tôi đặt lịch", _tone(220, 500))
    assert p > 0.7
