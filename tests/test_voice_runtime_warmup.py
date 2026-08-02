from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from soca.asr.context import ASRContextBuilder, ASRContextSourceRecord
from soca.asr.selection import ASRSelection
from soca.core.voice_runtime import (
    ResolvedVoiceRuntimeConfig,
    VoiceRuntimeBundle,
    VoiceRuntimeWarmupError,
    VoiceRuntimeWarmupResult,
    _smart_turn_model_dir,
    warm_up_voice_runtime,
)
from soca.tts import TTSResult


class FakeInnerASR:
    SAMPLING_RATE = 16000

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def transcribe(
        self,
        audio: np.ndarray,
        max_new_tokens: int = 128,
        *,
        context: str,
    ):
        assert context == ""
        self.calls.append((len(audio), max_new_tokens))
        return object()


class FakeRobustASR:
    confidence_guard_status = "disabled:test"

    def __init__(self) -> None:
        self.asr = FakeInnerASR()

    def snapshot_context(self):
        return ASRContextBuilder().build(())


class FakeContextAwareInnerASR:
    """Mirrors QwenASRBackend: accepts a context kwarg and exposes .context."""

    SAMPLING_RATE = 16000

    def __init__(self, context: str) -> None:
        self.context = context
        self.calls: list[dict] = []

    def transcribe(
        self, audio: np.ndarray, max_new_tokens: int = 128, *, context: str | None = None
    ):
        self.calls.append(
            {"audio_len": len(audio), "max_new_tokens": max_new_tokens, "context": context}
        )
        return object()


class FakeRobustASRWithContext:
    confidence_guard_status = "disabled:test"

    def __init__(self, inner: FakeContextAwareInnerASR) -> None:
        self.asr = inner

    def snapshot_context(self):
        records = (ASRContextSourceRecord(self.asr.context, "test"),) if self.asr.context else ()
        return ASRContextBuilder().build(records)


class FakeLLM:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def generate(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return object()


class FakeTTS:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(self, text: str, voice: str | None = None) -> TTSResult:
        self.calls.append(text)
        return TTSResult(
            text=text,
            audio=np.zeros(100, dtype=np.float32),
            sample_rate=24000,
            latency_ms=1.0,
            audio_duration_ms=1.0,
            rtf=1.0,
            voice=voice or "fake",
            engine="fake",
        )


class FakeTurnDetector:
    def __init__(self) -> None:
        self.warmups = 0

    def warmup(self) -> None:
        self.warmups += 1


class FailingTurnDetector:
    def warmup(self) -> None:
        raise RuntimeError("smart turn warmup failed")


def make_config() -> ResolvedVoiceRuntimeConfig:
    return ResolvedVoiceRuntimeConfig(
        profile_key="baseline",
        asr=ASRSelection.phowhisper("phowhisper_small"),
        llm_model="arcee_vylinh_3b_q4_k_m",
        tts_voice="NF",
        endpoint_silence_ms=700,
        adaptive_endpoint=False,
        max_record_ms=10_000,
        max_tokens=160,
        temperature=0.2,
        top_p=0.95,
        first_clause_enabled=True,
        first_clause_min_chars=12,
        first_clause_min_words=2,
        first_clause_max_scan_chars=80,
        pcm_crossfade_enabled=True,
        pcm_crossfade_ms=12.0,
        vault=Path("/tmp/soca-test-vault"),
        no_memory=True,
    )


def test_warm_up_voice_runtime_triggers_asr_llm_and_tts_first_call_paths() -> None:
    asr = FakeRobustASR()
    llm = FakeLLM()
    tts = FakeTTS()
    bundle = VoiceRuntimeBundle(
        config=make_config(),
        detector=object(),
        asr=asr,
        llm=llm,
        tts=tts,
        assistant_runtime=object(),
        pipeline=object(),
        memory_status="disabled:test",
        knowledge_status="disabled:test",
    )

    results = warm_up_voice_runtime(bundle, asr_seconds=0.5)

    assert [result.component for result in results] == ["asr", "llm", "tts"]
    assert all(result.ok for result in results)
    # Warm call (max_new_tokens=1) + a 3s representative probe that calibrates the
    # partial-caption cadence uses the runtime's selected partial budget.
    assert asr.asr.calls == [(8000, 1), (48000, 128)]
    assert bundle.partial_enabled is True
    assert bundle.partial_interval_ms == 800  # fake ASR ~instant -> clamps to floor
    assert llm.calls == [
        {
            "prompt": "xin chào",
            "max_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "inject_persona": True,
        }
    ]
    assert tts.calls == ["Xin chào, tôi là SoCa."]


def test_warm_up_voice_runtime_warms_both_partial_and_final_context_paths() -> None:
    """A context-aware backend must warm both the partial path
    (context="", cheap and used for every caption update) and the final
    path (the real context) — a cold context switch pays an extra prefill
    cost that must not land on the user's actual first turn."""
    inner = FakeContextAwareInnerASR(context="tech context")
    bundle = VoiceRuntimeBundle(
        config=make_config(),
        detector=object(),
        asr=FakeRobustASRWithContext(inner),
        llm=FakeLLM(),
        tts=FakeTTS(),
        assistant_runtime=object(),
        pipeline=object(),
        memory_status="disabled:test",
        knowledge_status="disabled:test",
    )

    warm_up_voice_runtime(bundle, asr_seconds=0.5)

    assert [call["context"] for call in inner.calls] == ["", "", "tech context"]
    # The final-context warm call only needs to pay the prefill cost, not
    # repeat a full representative decode (already measured by the second
    # call above) — must stay cheap (max_new_tokens=1).
    assert inner.calls[-1]["max_new_tokens"] == 1


def test_warm_up_voice_runtime_skips_final_context_warm_when_context_is_empty() -> None:
    inner = FakeContextAwareInnerASR(context="")
    bundle = VoiceRuntimeBundle(
        config=make_config(),
        detector=object(),
        asr=FakeRobustASRWithContext(inner),
        llm=FakeLLM(),
        tts=FakeTTS(),
        assistant_runtime=object(),
        pipeline=object(),
        memory_status="disabled:test",
        knowledge_status="disabled:test",
    )

    warm_up_voice_runtime(bundle, asr_seconds=0.5)

    assert [call["context"] for call in inner.calls] == ["", ""]


def test_warm_up_voice_runtime_includes_smart_turn_when_detector_exists() -> None:
    turn_detector = FakeTurnDetector()
    bundle = VoiceRuntimeBundle(
        config=make_config(),
        detector=object(),
        asr=FakeRobustASR(),
        llm=FakeLLM(),
        tts=FakeTTS(),
        assistant_runtime=object(),
        pipeline=object(),
        memory_status="disabled:test",
        knowledge_status="disabled:test",
        turn_detector=turn_detector,  # type: ignore[arg-type]
    )

    results = warm_up_voice_runtime(bundle, asr_seconds=0.5)

    assert [result.component for result in results] == ["asr", "llm", "tts", "smart_turn"]
    assert results[-1].ok is True
    assert turn_detector.warmups == 1


def test_warm_up_voice_runtime_fails_fast_when_smart_turn_warmup_fails() -> None:
    bundle = VoiceRuntimeBundle(
        config=make_config(),
        detector=object(),
        asr=FakeRobustASR(),
        llm=FakeLLM(),
        tts=FakeTTS(),
        assistant_runtime=object(),
        pipeline=object(),
        memory_status="disabled:test",
        knowledge_status="disabled:test",
        turn_detector=FailingTurnDetector(),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="smart turn warmup failed"):
        warm_up_voice_runtime(bundle, asr_seconds=0.5)


def test_voice_runtime_warmup_error_preserves_typed_failures() -> None:
    failure = VoiceRuntimeWarmupResult(
        component="asr",
        ok=False,
        latency_ms=12.0,
        detail="service timeout",
    )

    error = VoiceRuntimeWarmupError((failure,))

    assert error.failures == (failure,)
    assert str(error) == "Voice runtime warmup failed: asr: service timeout"

    with pytest.raises(ValueError, match="only failed"):
        VoiceRuntimeWarmupError(
            (
                VoiceRuntimeWarmupResult(
                    component="tts",
                    ok=True,
                    latency_ms=1.0,
                ),
            )
        )


def test_smart_turn_model_dir_points_to_repo_models_dir() -> None:
    assert _smart_turn_model_dir().as_posix().endswith("/models/smart-turn-v3-onnx")
