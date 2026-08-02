"""Regression tests for independent transcript quality checks."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from soca.asr.robust_asr import RobustASR
from soca.asr.vad import VADResult
from soca.asr.whisper_onnx import ASRResult, VietnameseASR

DUMMY_AUDIO = np.zeros(16000, dtype=np.float32)


class _FakeVAD:
    def detect(self, audio: np.ndarray) -> VADResult:
        duration_ms = len(audio) / 16_000 * 1000
        return VADResult(
            has_speech=True,
            speech_audio=audio,
            speech_duration_ms=duration_ms,
            original_duration_ms=duration_ms,
            speech_ratio=1.0,
            vad_latency_ms=0.0,
            n_speech_segments=1,
        )


class _NoLogprobASR:
    """Backend that cannot produce a real avg_logprob, e.g. a Qwen backend
    built with require_logprob=False. Mirrors QwenASRBackend's contract:
    `supports_avg_logprob = False` and a placeholder `avg_logprob = 0.0`.
    """

    supports_avg_logprob = False

    def __init__(self, text: str, model_key: str = "fake_no_logprob"):
        self.text = text
        self.model_key = model_key

    def transcribe(
        self,
        audio: np.ndarray,
        max_new_tokens: int = 128,
        *,
        context: str,
    ) -> ASRResult:
        del max_new_tokens, context
        return ASRResult(
            text=self.text,
            latency_ms=0.0,
            audio_duration_ms=len(audio) / 16_000 * 1000,
            rtf=0.0,
            avg_logprob=0.0,
        )


class _LogprobASR:
    """Backend with no `supports_avg_logprob` attribute at all, mirroring
    VietnameseASR: the pipeline must duck-type this as capable (default True)
    so existing behavior is unchanged.
    """

    def __init__(self, text: str, avg_logprob: float, model_key: str = "fake_logprob"):
        self.text = text
        self.avg_logprob = avg_logprob
        self.model_key = model_key

    def transcribe(
        self,
        audio: np.ndarray,
        max_new_tokens: int = 128,
        *,
        context: str,
    ) -> ASRResult:
        del max_new_tokens, context
        return ASRResult(
            text=self.text,
            latency_ms=0.0,
            audio_duration_ms=len(audio) / 16_000 * 1000,
            rtf=0.0,
            avg_logprob=self.avg_logprob,
        )


class _NoLogprobNoModelKeyASR:
    """Backend that has neither `supports_avg_logprob=True` nor `model_key`,
    e.g. a minimal custom backend. Used to check the no-logprob status string
    does not misreport an unrelated fallback model key.
    """

    supports_avg_logprob = False

    def __init__(self, text: str):
        self.text = text

    def transcribe(
        self,
        audio: np.ndarray,
        max_new_tokens: int = 128,
        *,
        context: str,
    ) -> ASRResult:
        del max_new_tokens, context
        return ASRResult(
            text=self.text,
            latency_ms=0.0,
            audio_duration_ms=len(audio) / 16_000 * 1000,
            rtf=0.0,
            avg_logprob=0.0,
        )


def test_vietnamese_asr_declares_logprob_capability_explicitly():
    # No instance needed (that would load ONNX models); the whole point is
    # this is a class-level declaration, not an inferred default.
    assert VietnameseASR.supports_avg_logprob is True


def test_missing_capability_attribute_is_logged_not_silent(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, logger="soca.asr.robust_asr"):
        RobustASR(
            asr=_LogprobASR("xin chào thế giới", avg_logprob=0.0),
            vad=_FakeVAD(),
        )

    assert any("supports_avg_logprob" in record.message for record in caplog.records)


def test_no_logprob_status_does_not_report_unrelated_default_model_key():
    pipeline = RobustASR(asr=_NoLogprobNoModelKeyASR("xin chào thế giới"), vad=_FakeVAD())

    assert "unknown_backend" in pipeline.confidence_guard_status
    # Must not silently claim this unrelated backend is phowhisper_tiny.
    assert "phowhisper_tiny" not in pipeline.confidence_guard_status


def test_compression_guard_status_stays_observable_when_logprob_guard_is_skipped():
    """The bug the split fixed: before this PR, a skipped logprob guard made
    `confidence_guard_status` read 'skipped:...' even on a run that was
    actually rejected by the (still-active) compression guard, making the
    status misleading. `compression_guard_status` must independently say the
    compression guard was enabled.
    """
    repeated = "xin chào " * 80
    pipeline = RobustASR(
        asr=_LogprobASR(repeated, avg_logprob=0.0, model_key="phowhisper_base"),
        vad=_FakeVAD(),
        max_compression_ratio=0.5,
        confidence_guard_skip_reason="skipped:missing_for_model:phowhisper_base",
    )

    result = pipeline.transcribe(DUMMY_AUDIO)

    assert result.rejection_reason.startswith("high_compression:")
    assert result.confidence_guard_status == "skipped:missing_for_model:phowhisper_base"
    assert result.compression_guard_status == "enabled"


def test_backend_without_logprob_still_rejects_high_compression():
    repeated = "xin chào " * 80
    pipeline = RobustASR(
        asr=_NoLogprobASR(repeated),
        vad=_FakeVAD(),
        max_compression_ratio=0.5,
    )

    result = pipeline.transcribe(DUMMY_AUDIO)

    assert result.text == ""
    assert result.rejection_reason.startswith("high_compression:")


def test_backend_without_logprob_reports_honest_status_and_keeps_clean_text():
    pipeline = RobustASR(asr=_NoLogprobASR("xin chào thế giới"), vad=_FakeVAD())

    assert pipeline.use_logprob_guard is False
    assert "backend_has_no_logprob" in pipeline.confidence_guard_status

    result = pipeline.transcribe(DUMMY_AUDIO)

    assert result.text == "xin chào thế giới"
    assert result.rejection_reason == ""
    assert "backend_has_no_logprob" in result.confidence_guard_status


def test_backend_without_supports_avg_logprob_attribute_defaults_to_capable():
    pipeline = RobustASR(
        asr=_LogprobASR("xin chào thế giới", avg_logprob=0.0, model_key="fake_logprob"),
        vad=_FakeVAD(),
        confidence_profile_model_key="fake_logprob",
    )

    assert pipeline.use_logprob_guard is True
    assert pipeline.confidence_guard_status == "enabled:fake_logprob"


def test_model_mismatch_skip_still_rejects_high_compression():
    """Behavior change: a skipped logprob guard (model mismatch, or an
    explicit missing-calibration reason) must not silently disable the
    compression guard too — compression is model-independent.
    """
    repeated = "xin chào " * 80
    pipeline = RobustASR(
        asr=_LogprobASR(repeated, avg_logprob=0.0, model_key="phowhisper_base"),
        vad=_FakeVAD(),
        max_compression_ratio=0.5,
        confidence_profile_model_key="phowhisper_tiny",  # mismatch vs runtime
    )

    assert pipeline.use_logprob_guard is False
    assert pipeline.use_compression_guard is True

    result = pipeline.transcribe(DUMMY_AUDIO)

    assert result.text == ""
    assert result.rejection_reason.startswith("high_compression:")
