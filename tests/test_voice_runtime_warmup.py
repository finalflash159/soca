from __future__ import annotations

from pathlib import Path

import numpy as np

from soca.core.voice_runtime import (
    ResolvedVoiceRuntimeConfig,
    VoiceRuntimeBundle,
    warm_up_voice_runtime,
)
from soca.tts import TTSResult


class FakeInnerASR:
    SAMPLING_RATE = 16000

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def transcribe(self, audio: np.ndarray, max_new_tokens: int = 128):
        self.calls.append((len(audio), max_new_tokens))
        return object()


class FakeRobustASR:
    boh_status = "disabled:test"
    confidence_guard_status = "disabled:test"

    def __init__(self) -> None:
        self.asr = FakeInnerASR()


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


def make_config() -> ResolvedVoiceRuntimeConfig:
    return ResolvedVoiceRuntimeConfig(
        profile_key="quality",
        asr_model="phowhisper_small",
        llm_model="arcee_vylinh_3b_q4_k_m",
        tts_model="omnivoice",
        tts_voice="emgai_dangiu",
        endpoint_silence_ms=700,
        max_record_ms=10_000,
        max_tokens=160,
        temperature=0.2,
        top_p=0.95,
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
    # partial-caption cadence (default max_new_tokens=128).
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
