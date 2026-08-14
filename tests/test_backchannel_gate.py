from __future__ import annotations

import numpy as np
import pytest

from eval.barge_in_replay import BargeInDecider
from soca.core.backchannel import (
    BackchannelClassificationError,
    BackchannelDecision,
    BargeInIntent,
    classify_barge_in_window,
)

_FRAME = 512
_SR = 16_000


class _Aec:
    def process(self, near: np.ndarray, far: np.ndarray) -> np.ndarray:
        del far
        return near


class _Vad:
    def __call__(self, frame: np.ndarray, sample_rate: int) -> float:
        del sample_rate
        return float(np.max(np.abs(frame)) > 0.1)


class _Classifier:
    def __init__(self, intents: list[BargeInIntent]) -> None:
        self.intents = intents
        self.windows: list[np.ndarray] = []

    def classify(self, audio: np.ndarray, sample_rate: int) -> BackchannelDecision:
        assert sample_rate == _SR
        self.windows.append(audio.copy())
        return BackchannelDecision(
            intent=self.intents.pop(0),
            confidence=0.95,
            model_id="fixture",
            model_revision="one",
            latency_ms=2.0,
        )


def test_backchannel_is_suppressed_but_later_interruption_still_fires() -> None:
    classifier = _Classifier([BargeInIntent.BACKCHANNEL, BargeInIntent.INTERRUPTION])
    speech = np.full(28 * _FRAME, 0.2, dtype=np.float32)
    silence = np.zeros(3 * _FRAME, dtype=np.float32)
    near = np.concatenate((speech[:13 * _FRAME], silence, speech[13 * _FRAME :]))

    result = BargeInDecider(
        aec=_Aec(),
        vad=_Vad(),
        backchannel_classifier=classifier,
    ).run(np.zeros_like(near), near)

    assert result.interrupted is True
    assert result.suppressed_backchannels == 1
    assert [item.intent for item in result.backchannel_decisions] == [
        BargeInIntent.BACKCHANNEL,
        BargeInIntent.INTERRUPTION,
    ]
    assert all(len(window) <= 1200 * _SR // 1000 for window in classifier.windows)


def test_no_classifier_preserves_explicit_acoustic_gate() -> None:
    near = np.full(13 * _FRAME, 0.2, dtype=np.float32)

    result = BargeInDecider(aec=_Aec(), vad=_Vad()).run(np.zeros_like(near), near)

    assert result.interrupted is True
    assert result.backchannel_decisions == ()


class _BrokenClassifier:
    def classify(self, audio: np.ndarray, sample_rate: int) -> BackchannelDecision:
        del audio, sample_rate
        raise RuntimeError("unavailable")


def test_classifier_failure_is_typed_and_never_falls_back_to_interrupt() -> None:
    near = np.full(13 * _FRAME, 0.2, dtype=np.float32)

    with pytest.raises(BackchannelClassificationError) as error:
        BargeInDecider(
            aec=_Aec(),
            vad=_Vad(),
            backchannel_classifier=_BrokenClassifier(),
        ).run(np.zeros_like(near), near)

    assert error.value.code == "classifier_unavailable"


def test_classifier_contract_rejects_empty_audio() -> None:
    with pytest.raises(BackchannelClassificationError, match="invalid_audio_window"):
        classify_barge_in_window(_BrokenClassifier(), np.array([], dtype=np.float32), _SR)
