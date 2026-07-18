from __future__ import annotations

import numpy as np
import pytest

from soca.core.turn_taking import (
    PARTIAL_FLOOR_MS,
    SPEECH_CONFIRM_FRAMES,
    VAD_FRAME_SAMPLES,
    IncrementalVadTracker,
    LocalAgreement,
    is_incomplete_vietnamese,
    partial_interval_from_cost,
    required_silence_ms,
)


class FakeSilero:
    """Scripted per-frame probabilities; mimics Silero's call signature."""

    def __init__(self, probs: list[float]):
        self.probs = list(probs)
        self.resets = 0

    def __call__(self, tensor, sample_rate):
        class _Out:
            def __init__(self, v):
                self._v = v

            def item(self):
                return self._v

        return _Out(self.probs.pop(0) if self.probs else 0.0)

    def reset_states(self):
        self.resets += 1


def frames(n: int) -> np.ndarray:
    return np.zeros(n * VAD_FRAME_SAMPLES, dtype=np.float32)


class Cfg:
    endpoint_silence_ms = 700
    adaptive = True
    patient_silence_ms = 1100
    eager_silence_ms = 450
    short_speech_ms = 1200
    long_speech_ms = 4000
    incomplete_hold_ms = 350


def test_policy_patient_eager_and_interpolation():
    cfg = Cfg()
    assert required_silence_ms(500, cfg) == 1100          # short -> patient
    assert required_silence_ms(9000, cfg) == 450          # long -> cut fast
    assert required_silence_ms(2600, cfg) == pytest.approx(775)  # middle -> interpolate
    assert required_silence_ms(500, cfg, incomplete=True) == 1450
    cfg.adaptive = False
    assert required_silence_ms(500, cfg) == 700           # legacy behavior


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("tôi muốn nói với", True),      # "với" is a connective, sentence still open
        ("cho mình hỏi,", True),
        ("tức là", True),
        ("ờ", True),
        ("mấy giờ rồi", False),          # trap: "rồi" ends a complete sentence
        ("cái này hay", False),          # trap: "hay" ends a complete sentence
        ("bật đèn lên.", False),
        ("", False),
    ],
)
def test_incomplete_vietnamese(text, expected):
    assert is_incomplete_vietnamese(text) is expected


def test_tracker_confirms_speech_then_counts_silence():
    # 3 frame speech (confirm) + 2 frame im -> speech 96ms, silence 64ms
    model = FakeSilero([0.9, 0.9, 0.9, 0.1, 0.1])
    tracker = IncrementalVadTracker(model)
    tracker.feed(frames(5))
    assert tracker.has_speech is True
    assert tracker.speech_ms == pytest.approx(32.0 * SPEECH_CONFIRM_FRAMES)
    assert tracker.silence_run_ms == pytest.approx(64.0)


def test_tracker_ignores_short_blip():
    # 2 "speech" frames (below the confirm threshold) amid silence -> no speech
    model = FakeSilero([0.1, 0.9, 0.9, 0.1])
    tracker = IncrementalVadTracker(model)
    tracker.feed(frames(4))
    assert tracker.has_speech is False
    assert tracker.silence_run_ms == 0.0   # no speech yet -> silence not counted


def test_tracker_buffers_residual_and_resets():
    model = FakeSilero([0.9] * 10)
    tracker = IncrementalVadTracker(model)
    tracker.feed(np.zeros(1600, dtype=np.float32))   # 3 frame + residual 64
    assert len(tracker._residual) == 1600 - 3 * VAD_FRAME_SAMPLES
    tracker.reset()
    assert model.resets == 1 and tracker.speech_ms == 0.0


def test_local_agreement_commits_common_prefix_only():
    la = LocalAgreement()
    assert la.update("cho mình hỏi") == ("", "cho mình hỏi")
    committed, tentative = la.update("cho mình hỏi về lịch")
    assert committed == "cho mình hỏi"
    assert tentative == "về lịch"
    committed, _ = la.update("cho mình hỏi vệ sinh")   # tail changed its mind
    assert committed == "cho mình hỏi"                 # committed never shrinks


def test_partial_interval_from_cost():
    # fast + many cores -> hits floor
    assert partial_interval_from_cost(120, cpu_count=8) == (PARTIAL_FLOOR_MS, True)
    # moderate per_call, enough cores -> 2x (margin)
    assert partial_interval_from_cost(500, cpu_count=8) == (1000, True)
    # slow device -> disable partial
    _interval, enabled = partial_interval_from_cost(2000, cpu_count=2)
    assert enabled is False
    # fewer cores -> rho scales up -> higher interval for same per_call
    assert (
        partial_interval_from_cost(300, cpu_count=2)[0]
        > partial_interval_from_cost(300, cpu_count=8)[0]
    )
