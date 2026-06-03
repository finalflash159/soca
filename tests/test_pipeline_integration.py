from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soca.core import NullAudioPlayer, VoicePipeline
from soca.tts import TTSResult


@dataclass(frozen=True)
class FakeASRResult:
    text: str
    rejection_reason: str = ""


@dataclass(frozen=True)
class FakeLLMResult:
    text: str


class FakeASR:
    def __init__(self, text: str, rejection_reason: str = "") -> None:
        self.text = text
        self.rejection_reason = rejection_reason
        self.calls = 0

    def transcribe(self, audio: np.ndarray) -> FakeASRResult:
        self.calls += 1
        return FakeASRResult(text=self.text, rejection_reason=self.rejection_reason)


class SpyLLM:
    def __init__(self, response: str = "Xin chao tu LLM") -> None:
        self.response = response
        self.calls: list[str] = []

    def generate(self, text: str) -> FakeLLMResult:
        self.calls.append(text)
        return FakeLLMResult(text=self.response)

    def generate_stream(self, text: str, max_tokens: int = 128, temperature: float = 0.0):
        self.calls.append(text)
        yield from ["Xin chào bạn. ", "Mình là SoCa."]


class SpyTTS:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(self, text: str, voice: str | None = None) -> TTSResult:
        self.calls.append(text)
        return TTSResult(
            text=text,
            audio=np.zeros(2400, dtype=np.float32),
            sample_rate=24000,
            latency_ms=10.0,
            audio_duration_ms=100.0,
            rtf=0.1,
            voice=voice or "NF",
            engine="fake",
        )


class FailingTTS:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(self, text: str, voice: str | None = None) -> TTSResult:
        self.calls.append(text)
        raise RuntimeError("tts failed")


def test_voice_pipeline_happy_path_calls_llm_and_tts() -> None:
    asr = FakeASR("xin chao")
    llm = SpyLLM(response="chao ban")
    tts = SpyTTS()
    pipeline = VoicePipeline(asr=asr, llm=llm, tts=tts)

    result = pipeline.turn(np.zeros(16000, dtype=np.float32))

    assert result.rejected is False
    assert result.transcript == "xin chao"
    assert result.response_text == "chao ban"
    assert result.rejection_reason == ""
    assert result.tts is not None
    assert result.tts.text == "chao ban"
    assert asr.calls == 1
    assert llm.calls == ["xin chao"]
    assert tts.calls == ["chao ban"]
    assert set(result.stage_latencies_ms) == {"asr", "llm", "tts"}
    assert result.total_latency_ms >= sum(result.stage_latencies_ms.values())


def test_voice_pipeline_strips_markdown_before_tts_but_keeps_response_text() -> None:
    asr = FakeASR("tình hình")
    llm = SpyLLM(response="**Tình hình:** nên ăn **đủ đạm**.")
    tts = SpyTTS()
    pipeline = VoicePipeline(asr=asr, llm=llm, tts=tts)

    result = pipeline.turn(np.zeros(16000, dtype=np.float32))

    assert result.response_text == "**Tình hình:** nên ăn **đủ đạm**."
    assert tts.calls == ["Tình hình: nên ăn đủ đạm."]


def test_voice_pipeline_reject_path_skips_llm_and_tts() -> None:
    asr = FakeASR("", rejection_reason="no_speech")
    llm = SpyLLM()
    tts = SpyTTS()
    pipeline = VoicePipeline(asr=asr, llm=llm, tts=tts)

    result = pipeline.turn(np.zeros(16000, dtype=np.float32))

    assert result.rejected is True
    assert result.transcript == ""
    assert result.response_text == pipeline.reject_response
    assert result.rejection_reason == "no_speech"
    assert result.tts is None
    assert asr.calls == 1
    assert llm.calls == []
    assert tts.calls == []
    assert set(result.stage_latencies_ms) == {"asr"}


def test_voice_pipeline_empty_transcript_uses_default_rejection_reason() -> None:
    pipeline = VoicePipeline(asr=FakeASR("   "), llm=SpyLLM(), tts=SpyTTS())

    result = pipeline.turn(np.zeros(16000, dtype=np.float32))

    assert result.rejected is True
    assert result.rejection_reason == "empty_transcript"


def test_voice_pipeline_resets_metrics_between_turns() -> None:
    pipeline = VoicePipeline(asr=FakeASR("xin chao"), llm=SpyLLM(), tts=SpyTTS())

    first = pipeline.turn(np.zeros(16000, dtype=np.float32))
    second = pipeline.turn(np.zeros(16000, dtype=np.float32))

    assert set(first.stage_latencies_ms) == {"asr", "llm", "tts"}
    assert set(second.stage_latencies_ms) == {"asr", "llm", "tts"}


def test_voice_pipeline_streaming_yields_asr_tokens_sentences_audio_and_done():
    asr = FakeASR("xin chao")
    llm = SpyLLM(response="unused")
    tts = SpyTTS()
    pipeline = VoicePipeline(asr=asr, llm=llm, tts=tts)

    events = list(
        pipeline.turn_streaming(
            np.zeros(16000, dtype=np.float32),
            audio_sink=NullAudioPlayer(),
            min_sentence_chars=8,
        )
    )

    event_types = [event.type for event in events]

    assert "asr" in event_types
    assert "llm_token" in event_types
    assert "sentence" in event_types
    assert "tts" in event_types
    assert "audio" in event_types
    assert event_types[-1] == "done"
    assert asr.calls == 1
    assert llm.calls == ["xin chao"]
    assert tts.calls == ["Xin chào bạn.", "Mình là SoCa."]


def test_voice_pipeline_streaming_strips_markdown_before_tts():
    asr = FakeASR("tình hình")
    llm = SpyLLM(response="unused")
    llm.generate_stream = lambda text, max_tokens=128, temperature=0.0: iter(
        ["**Tình hình:** nên ăn **đủ đạm**."]
    )
    tts = SpyTTS()
    pipeline = VoicePipeline(asr=asr, llm=llm, tts=tts)

    events = list(
        pipeline.turn_streaming(
            np.zeros(16000, dtype=np.float32),
            audio_sink=NullAudioPlayer(),
            min_sentence_chars=8,
        )
    )

    assert any(event.type == "sentence" and "**Tình hình:**" in event.text for event in events)
    assert any(event.type == "tts" and event.text == "Tình hình: nên ăn đủ đạm." for event in events)
    assert tts.calls == ["Tình hình: nên ăn đủ đạm."]


def test_voice_pipeline_streaming_reject_path_speaks_fallback_by_default():
    asr = FakeASR("", rejection_reason="no_speech")
    llm = SpyLLM()
    tts = SpyTTS()
    pipeline = VoicePipeline(asr=asr, llm=llm, tts=tts)

    events = list(
        pipeline.turn_streaming(
            np.zeros(16000, dtype=np.float32),
            audio_sink=NullAudioPlayer(),
        )
    )

    assert [event.type for event in events] == [
        "asr",
        "repair",
        "sentence",
        "tts",
        "audio",
        "done",
    ]
    repair = next(event for event in events if event.type == "repair")
    assert repair.metadata["repair_kind"] == "no_input"
    assert events[-1].metadata["rejected"] is True
    assert events[-1].metadata["repair_kind"] == "no_input"
    assert llm.calls == []
    assert tts.calls == [pipeline.reject_response]


def test_voice_pipeline_streaming_reject_path_can_skip_fallback_speech():
    asr = FakeASR("", rejection_reason="no_speech")
    llm = SpyLLM()
    tts = SpyTTS()
    pipeline = VoicePipeline(asr=asr, llm=llm, tts=tts)

    events = list(
        pipeline.turn_streaming(
            np.zeros(16000, dtype=np.float32),
            audio_sink=NullAudioPlayer(),
            speak_rejections=False,
        )
    )

    assert [event.type for event in events] == ["asr", "repair", "done"]
    assert events[-1].metadata["rejected"] is True
    assert llm.calls == []
    assert tts.calls == []


def test_voice_pipeline_streaming_yields_error_when_tts_fails():
    asr = FakeASR("xin chao")
    llm = SpyLLM(response="unused")
    tts = FailingTTS()
    pipeline = VoicePipeline(asr=asr, llm=llm, tts=tts)

    events = list(
        pipeline.turn_streaming(
            np.zeros(16000, dtype=np.float32),
            audio_sink=NullAudioPlayer(),
            min_sentence_chars=8,
        )
    )

    error_events = [event for event in events if event.type == "error"]

    assert error_events
    assert error_events[0].text == "tts failed"
    assert events[-1].type == "done"
    assert tts.calls == ["Xin chào bạn."]
