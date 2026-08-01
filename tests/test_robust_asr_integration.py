"""Integration-style tests for RobustASR using dependency injection.

These tests intentionally avoid loading PhoWhisper or Silero. They verify
pipeline behavior and rejection ordering with small mocks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from soca.asr.robust_asr import RobustASR, load_confidence_guard_calibration
from soca.asr.vad import VADResult
from soca.asr.whisper_onnx import ASRResult

DUMMY_AUDIO = np.zeros(16000, dtype=np.float32)


class MockASR:
    def __init__(
        self,
        text: str,
        avg_logprob: float = 0.0,
        model_key: str = "phowhisper_tiny",
        alternatives: tuple[str, ...] = (),
        context: str = "",
    ):
        self.text = text
        self.avg_logprob = avg_logprob
        self.model_key = model_key
        self.alternatives = alternatives
        self.context = context
        self.calls = 0

    def transcribe(self, audio: np.ndarray) -> ASRResult:
        self.calls += 1
        return ASRResult(
            text=self.text,
            latency_ms=0.0,
            audio_duration_ms=len(audio) / 16_000 * 1000,
            rtf=0.0,
            avg_logprob=self.avg_logprob,
            alternatives=self.alternatives,
        )


class MockVAD:
    def __init__(self, has_speech: bool = True):
        self.has_speech = has_speech

    def detect(self, audio: np.ndarray) -> VADResult:
        speech_audio = audio if self.has_speech else np.array([], dtype=np.float32)
        duration_ms = len(audio) / 16_000 * 1000
        speech_duration_ms = len(speech_audio) / 16_000 * 1000
        return VADResult(
            has_speech=self.has_speech,
            speech_audio=speech_audio,
            speech_duration_ms=speech_duration_ms,
            original_duration_ms=duration_ms,
            speech_ratio=1.0 if self.has_speech else 0.0,
            vad_latency_ms=0.0,
            n_speech_segments=1 if self.has_speech else 0,
        )


def test_no_speech_skips_asr():
    asr = MockASR("xin chào", avg_logprob=0.0)
    pipeline = RobustASR(asr=asr, vad=MockVAD(has_speech=False))

    result = pipeline.transcribe(DUMMY_AUDIO)

    assert result.text == ""
    assert result.rejection_reason == "no_speech"
    assert result.has_speech is False
    assert result.asr is None
    assert asr.calls == 0


def test_low_confidence_rejected_before_text_filters():
    pipeline = RobustASR(
        asr=MockASR("thôi.", avg_logprob=-0.90),
        vad=MockVAD(),
        min_avg_logprob=-0.25,
    )

    result = pipeline.transcribe(DUMMY_AUDIO)

    assert result.text == ""
    assert result.rejection_reason == "low_confidence:-0.90"
    assert result.text_after_deloop == "thôi."


def test_high_compression_rejected():
    repeated = "xin chào " * 80
    pipeline = RobustASR(
        asr=MockASR(repeated, avg_logprob=0.0),
        vad=MockVAD(),
        max_compression_ratio=0.5,
    )

    result = pipeline.transcribe(DUMMY_AUDIO)

    assert result.text == ""
    assert result.rejection_reason.startswith("high_compression:")
    assert result.compression_ratio > 0.5


def test_context_echo_rejected():
    """§Q1b.3: a context-aware backend can return its system prompt verbatim
    instead of transcribing the audio. That text passes every other check
    (clean grammar, no repetition, normal compression), so it needs its own
    dedicated rejection reason."""
    context = (
        "Cuộc hội thoại về lập trình. GitHub, PyTorch, TensorFlow, "
        "PostgreSQL, Docker, Kubernetes."
    )
    pipeline = RobustASR(
        asr=MockASR(context, avg_logprob=0.0, context=context),
        vad=MockVAD(),
    )

    result = pipeline.transcribe(DUMMY_AUDIO)

    assert result.text == ""
    assert result.rejection_reason == "context_echo"


def test_short_sentence_sharing_context_vocabulary_is_not_rejected():
    context = (
        "Cuộc hội thoại về lập trình. GitHub, PyTorch, TensorFlow, "
        "PostgreSQL, Docker, Kubernetes."
    )
    phrase = "mở repo trên github"
    pipeline = RobustASR(
        asr=MockASR(phrase, avg_logprob=0.0, context=context),
        vad=MockVAD(),
    )

    result = pipeline.transcribe(DUMMY_AUDIO)

    assert result.text == phrase
    assert result.rejection_reason == ""


def test_context_echo_check_is_skipped_when_backend_has_no_context():
    """A backend without a .context attribute (VietnameseASR) must not be
    affected by this guard at all — duck-typed default, not an error."""
    pipeline = RobustASR(
        asr=MockASR("xin chào thế giới", avg_logprob=0.0),
        vad=MockVAD(),
    )

    result = pipeline.transcribe(DUMMY_AUDIO)

    assert result.text == "xin chào thế giới"
    assert result.rejection_reason == ""


def test_deloop_then_clean_passes():
    pipeline = RobustASR(
        asr=MockASR("xin chào xin chào", avg_logprob=0.0),
        vad=MockVAD(),
    )

    result = pipeline.transcribe(DUMMY_AUDIO)

    assert result.text == "xin chào"
    assert result.text_after_deloop == "xin chào"
    assert result.was_looping is True
    assert result.rejection_reason == ""


def test_backend_alternatives_survive_robust_asr_without_domain_rewrite():
    alternatives = ("định lý Bayes", "định lý bài ét")
    pipeline = RobustASR(
        asr=MockASR("định lý bày ét", alternatives=alternatives),
        vad=MockVAD(),
    )

    result = pipeline.transcribe(DUMMY_AUDIO)

    assert result.alternatives == alternatives


def test_boh_like_phrase_is_preserved_in_production_transcript():
    phrase = "cảm ơn các bạn đã xem video"
    pipeline = RobustASR(
        asr=MockASR(phrase, avg_logprob=0.0),
        vad=MockVAD(),
    )

    result = pipeline.transcribe(DUMMY_AUDIO)

    assert result.text == phrase
    assert result.rejection_reason == ""


def test_filler_only_rejected_by_heuristic():
    pipeline = RobustASR(
        asr=MockASR("ờ ừm", avg_logprob=0.0),
        vad=MockVAD(),
    )

    result = pipeline.transcribe(DUMMY_AUDIO)

    assert result.text == ""
    assert result.rejection_reason == "heuristic:filler_only"


def test_unprofiled_asr_model_skips_tiny_confidence_guard():
    pipeline = RobustASR(
        asr=MockASR("xin chào thế giới", avg_logprob=-0.90, model_key="phowhisper_base"),
        vad=MockVAD(),
        min_avg_logprob=-0.25,
    )

    result = pipeline.transcribe(DUMMY_AUDIO)

    assert result.text == "xin chào thế giới"
    assert result.rejection_reason == ""
    assert result.confidence_guard_status == (
        "skipped:model_mismatch:profile=phowhisper_tiny,runtime=phowhisper_base"
    )


def test_matching_confidence_profile_keeps_guard_enabled_for_custom_model():
    pipeline = RobustASR(
        asr=MockASR("xin chào thế giới", avg_logprob=-0.90, model_key="phowhisper_base"),
        vad=MockVAD(),
        min_avg_logprob=-0.25,
        confidence_profile_model_key="phowhisper_base",
    )

    result = pipeline.transcribe(DUMMY_AUDIO)

    assert result.text == ""
    assert result.rejection_reason == "low_confidence:-0.90"
    assert result.confidence_guard_status == "enabled:phowhisper_base"


def test_loads_model_specific_confidence_calibration(tmp_path: Path):
    path = tmp_path / "threshold_calibration.json"
    path.write_text(
        json.dumps(
            {
                "asr_confidence_by_model": {
                    "phowhisper_base": {
                        "model_key": "phowhisper_base",
                        "recommended_thresholds": {
                            "min_avg_logprob": -0.5,
                            "max_compression_ratio": 2.7,
                        },
                        "created_at_utc": "2026-06-03T00:00:00+00:00",
                    }
                },
                "asr_confidence": {
                    "model_key": "phowhisper_tiny",
                    "recommended_thresholds": {
                        "min_avg_logprob": -0.725,
                        "max_compression_ratio": 2.4,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    calibration = load_confidence_guard_calibration("phowhisper_base", path)

    assert calibration is not None
    assert calibration.model_key == "phowhisper_base"
    assert calibration.min_avg_logprob == -0.5
    assert calibration.max_compression_ratio == 2.7


def test_missing_model_specific_confidence_calibration_returns_none(tmp_path: Path):
    path = tmp_path / "threshold_calibration.json"
    path.write_text(
        json.dumps(
            {
                "asr_confidence": {
                    "model_key": "phowhisper_tiny",
                    "recommended_thresholds": {
                        "min_avg_logprob": -0.725,
                        "max_compression_ratio": 2.4,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    assert load_confidence_guard_calibration("phowhisper_small", path) is None


def test_explicit_missing_confidence_calibration_disables_guard():
    pipeline = RobustASR(
        asr=MockASR("xin chào thế giới", avg_logprob=-0.90, model_key="phowhisper_base"),
        vad=MockVAD(),
        min_avg_logprob=-0.25,
        confidence_guard_skip_reason="skipped:missing_for_model:phowhisper_base",
    )

    result = pipeline.transcribe(DUMMY_AUDIO)

    assert result.text == "xin chào thế giới"
    assert result.rejection_reason == ""
    assert result.confidence_guard_status == "skipped:missing_for_model:phowhisper_base"
