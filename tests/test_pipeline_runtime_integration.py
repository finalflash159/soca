from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soca.core import NullAudioPlayer, RuntimeResult, RuntimeRoute, RuntimeTrace, VoicePipeline
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
    def __init__(self) -> None:
        self.generate_calls: list[str] = []
        self.stream_calls: list[str] = []

    def generate(self, text: str) -> FakeLLMResult:
        self.generate_calls.append(text)
        return FakeLLMResult("legacy response")

    def generate_stream(self, text: str):
        self.stream_calls.append(text)
        yield "legacy stream"


class SpyRuntime:
    def __init__(
        self,
        response_text: str = "Câu trả lời runtime.",
        route: RuntimeRoute = RuntimeRoute.TOOL_DIRECT,
        blocked: bool = False,
        used_tool: bool = True,
        used_llm: bool = False,
    ) -> None:
        self.response_text = response_text
        self.route = route
        self.blocked = blocked
        self.used_tool = used_tool
        self.used_llm = used_llm
        self.calls: list[dict] = []

    def run_text_turn(self, text: str, *, source: str = "text", metadata=None) -> RuntimeResult:
        self.calls.append(
            {
                "text": text,
                "source": source,
                "metadata": metadata or {},
            }
        )
        llm_result = FakeLLMResult("runtime llm result") if self.used_llm else None
        trace = RuntimeTrace(
            route=self.route,
            used_tool=self.used_tool,
            used_llm=self.used_llm,
            blocked=self.blocked,
        )
        return RuntimeResult(
            response_text=self.response_text,
            route=self.route,
            blocked=self.blocked,
            trace=trace,
            llm_result=llm_result,
        )


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


def test_turn_delegates_transcript_to_assistant_runtime_instead_of_llm() -> None:
    asr = FakeASR("mấy giờ rồi")
    llm = SpyLLM()
    tts = SpyTTS()
    runtime = SpyRuntime("Bây giờ là 09:30.")
    pipeline = VoicePipeline(asr=asr, llm=llm, tts=tts, assistant_runtime=runtime)

    result = pipeline.turn(np.zeros(16000, dtype=np.float32))

    assert result.rejected is False
    assert result.transcript == "mấy giờ rồi"
    assert result.response_text == "Bây giờ là 09:30."
    assert result.runtime_result is not None
    assert result.runtime_result.route == RuntimeRoute.TOOL_DIRECT
    assert runtime.calls == [
        {
            "text": "mấy giờ rồi",
            "source": "asr",
            "metadata": {"asr_rejection_reason": ""},
        }
    ]
    assert llm.generate_calls == []
    assert tts.calls == ["Bây giờ là 09:30."]
    assert set(result.stage_latencies_ms) == {"asr", "runtime", "tts"}


def test_turn_empty_transcript_skips_assistant_runtime() -> None:
    asr = FakeASR("", rejection_reason="no_speech")
    llm = SpyLLM()
    tts = SpyTTS()
    runtime = SpyRuntime()
    pipeline = VoicePipeline(asr=asr, llm=llm, tts=tts, assistant_runtime=runtime)

    result = pipeline.turn(np.zeros(16000, dtype=np.float32))

    assert result.rejected is True
    assert result.runtime_result is None
    assert runtime.calls == []
    assert llm.generate_calls == []
    assert tts.calls == []


def test_turn_keeps_llm_result_from_runtime_when_runtime_used_llm() -> None:
    runtime = SpyRuntime(
        response_text="LLM fallback.",
        route=RuntimeRoute.LLM_FALLBACK,
        used_tool=False,
        used_llm=True,
    )
    pipeline = VoicePipeline(
        asr=FakeASR("xin chào"),
        llm=SpyLLM(),
        tts=SpyTTS(),
        assistant_runtime=runtime,
    )

    result = pipeline.turn(np.zeros(16000, dtype=np.float32))

    assert result.llm_result is not None
    assert result.runtime_result is not None
    assert result.runtime_result.trace.used_llm is True


def test_streaming_runtime_path_emits_runtime_event_and_tts_without_llm_stream() -> None:
    asr = FakeASR("wiki chất đạm")
    llm = SpyLLM()
    tts = SpyTTS()
    runtime = SpyRuntime("Chất đạm hỗ trợ cơ bắp. Nên ăn vừa đủ.")
    pipeline = VoicePipeline(asr=asr, llm=llm, tts=tts, assistant_runtime=runtime)

    events = list(
        pipeline.turn_streaming(
            np.zeros(16000, dtype=np.float32),
            audio_sink=NullAudioPlayer(),
            min_sentence_chars=8,
        )
    )

    event_types = [event.type for event in events]

    assert event_types[0] == "asr"
    assert "runtime" in event_types
    assert "llm_token" not in event_types
    assert "sentence" in event_types
    assert "tts" in event_types
    assert "audio" in event_types
    assert event_types[-1] == "done"
    assert llm.stream_calls == []
    assert tts.calls == ["Chất đạm hỗ trợ cơ bắp.", "Nên ăn vừa đủ."]
    assert events[-1].metadata["runtime_route"] == RuntimeRoute.TOOL_DIRECT.value


def test_streaming_runtime_blocked_response_is_still_spoken_safely() -> None:
    runtime = SpyRuntime(
        response_text="Mình không thể xử lý yêu cầu này một cách an toàn.",
        route=RuntimeRoute.BLOCKED,
        blocked=True,
        used_tool=False,
    )
    tts = SpyTTS()
    pipeline = VoicePipeline(
        asr=FakeASR("đọc private/secrets.md"),
        llm=SpyLLM(),
        tts=tts,
        assistant_runtime=runtime,
    )

    events = list(
        pipeline.turn_streaming(
            np.zeros(16000, dtype=np.float32),
            audio_sink=NullAudioPlayer(),
        )
    )

    assert tts.calls == ["Mình không thể xử lý yêu cầu này một cách an toàn."]
    assert events[-1].metadata["runtime_blocked"] is True
    assert events[-1].metadata["rejected"] is False

