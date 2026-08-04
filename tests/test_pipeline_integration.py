from __future__ import annotations

import dataclasses
import threading
from dataclasses import dataclass

import numpy as np
import pytest

from soca.core import NullAudioPlayer, VoicePipeline
from soca.knowledge import KnowledgeCitation
from soca.tts import TTSResult


@dataclass(frozen=True)
class FakeASRResult:
    text: str
    rejection_reason: str = ""
    alternatives: tuple[str, ...] = ()
    context_digest: str = ""
    context_provenance: tuple[str, ...] = ()


@dataclass(frozen=True)
class FakeLLMResult:
    text: str


class FakeASR:
    def __init__(
        self,
        text: str,
        rejection_reason: str = "",
        alternatives: tuple[str, ...] = (),
        context_digest: str = "",
        context_provenance: tuple[str, ...] = (),
    ) -> None:
        self.text = text
        self.rejection_reason = rejection_reason
        self.alternatives = alternatives
        self.context_digest = context_digest
        self.context_provenance = context_provenance
        self.calls = 0

    def transcribe(self, audio: np.ndarray) -> FakeASRResult:
        self.calls += 1
        return FakeASRResult(
            text=self.text,
            rejection_reason=self.rejection_reason,
            alternatives=self.alternatives,
            context_digest=self.context_digest,
            context_provenance=self.context_provenance,
        )


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


class BrokenRateTTS(SpyTTS):
    """TTS engine that reports an unusable sample rate."""

    def synthesize(self, text: str, voice: str | None = None) -> TTSResult:
        result = super().synthesize(text, voice)
        return dataclasses.replace(result, sample_rate=0)


class FailingTTS:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(self, text: str, voice: str | None = None) -> TTSResult:
        self.calls.append(text)
        raise RuntimeError("tts failed")


@dataclass(frozen=True)
class FakeRuntimeResult:
    response_text: str
    route: object = None
    blocked: bool = False
    trace: object = None
    citations: tuple = ()
    usage: object = None


@dataclass(frozen=True)
class FakeRuntimeStreamEvent:
    type: str
    text: str = ""
    result: object = None


class FakeStreamingRuntime:
    """Minimal AssistantRuntime stand-in exposing stream_text_turn.

    Streams each sentence as a token+sentence event, then a final result event,
    matching what VoicePipeline._turn_streaming_runtime_stream consumes.
    """

    def __init__(
        self,
        sentences: list[str],
        *,
        citations: tuple[KnowledgeCitation, ...] = (),
    ) -> None:
        self.sentences = list(sentences)
        self.citations = citations
        self.calls: list[str] = []
        self.metadata_calls: list[dict | None] = []

    def stream_text_turn(
        self,
        text: str,
        *,
        source: str = "text",
        metadata: dict | None = None,
        min_sentence_chars: int = 24,
        first_sentence_min_chars: int | None = None,
        first_clause_enabled: bool = True,
        first_clause_min_chars: int = 12,
        first_clause_min_words: int = 2,
        first_clause_max_scan_chars: int = 80,
    ):
        self.calls.append(text)
        self.metadata_calls.append(metadata)
        for sentence in self.sentences:
            yield FakeRuntimeStreamEvent(type="token", text=sentence)
            yield FakeRuntimeStreamEvent(type="sentence", text=sentence)
        yield FakeRuntimeStreamEvent(
            type="result",
            result=FakeRuntimeResult(
                response_text=" ".join(self.sentences),
                citations=self.citations,
            ),
        )


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
    assert result.response_text
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
    runtime = FakeStreamingRuntime(["Xin chào bạn.", "Mình là SoCa."])
    tts = SpyTTS()
    pipeline = VoicePipeline(asr=asr, llm=SpyLLM(), tts=tts, assistant_runtime=runtime)

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
    assert "playback_started" in event_types
    assert "audio" in event_types
    assert event_types[-1] == "done"
    assert asr.calls == 1
    assert runtime.calls == ["xin chao"]
    assert tts.calls == ["Xin chào bạn.", "Mình là SoCa."]
    assert all(
        event.metadata == {"delivery": "answer_delta", "terminal": False}
        for event in events
        if event.type == "sentence"
    )
    assert all(
        event.metadata["delivery"] == "final"
        for event in events
        if event.type == "tts"
    )
    for playback_started in (
        event for event in events if event.type == "playback_started"
    ):
        chunk_index = playback_started.metadata["chunk_index"]
        tts_index = next(
            index
            for index, event in enumerate(events)
            if event.type == "tts" and event.metadata["chunk_index"] == chunk_index
        )
        playback_index = events.index(playback_started)
        audio_index = next(
            index
            for index, event in enumerate(events)
            if event.type == "audio" and event.metadata["chunk_index"] == chunk_index
        )
        assert tts_index < playback_index < audio_index
        assert playback_started.metadata["audio_duration_ms"] == pytest.approx(100.0)
        assert playback_started.metadata["sync_granularity"] == "audio_chunk"


def test_playback_started_omits_duration_when_sample_rate_is_unusable() -> None:
    asr = FakeASR("xin chao")
    runtime = FakeStreamingRuntime(["Xin chào bạn."])
    pipeline = VoicePipeline(
        asr=asr,
        llm=SpyLLM(),
        tts=BrokenRateTTS(),
        assistant_runtime=runtime,
    )

    events = list(
        pipeline.turn_streaming(
            np.zeros(16000, dtype=np.float32),
            audio_sink=NullAudioPlayer(),
            min_sentence_chars=8,
        )
    )

    playback_started = [event for event in events if event.type == "playback_started"]
    assert playback_started, "playback still has to be announced"
    for event in playback_started:
        # Unknown duration is reported as absent, never as a fabricated 0.0.
        assert "audio_duration_ms" not in event.metadata
        assert event.metadata["sync_granularity"] == "audio_chunk"
    assert [event.type for event in events][-1] == "done"


def test_voice_pipeline_forwards_asr_alternatives_for_goal_repair() -> None:
    alternatives = ("định lý Bayes", "định lý bài ét")
    asr = FakeASR("định lý bày ét", alternatives=alternatives)
    runtime = FakeStreamingRuntime(["Mình sẽ kiểm tra ghi chú của bạn."])
    pipeline = VoicePipeline(asr=asr, llm=SpyLLM(), tts=SpyTTS(), assistant_runtime=runtime)

    events = list(
        pipeline.turn_streaming(
            np.zeros(16000, dtype=np.float32),
            audio_sink=NullAudioPlayer(),
            min_sentence_chars=8,
        )
    )

    asr_event = next(event for event in events if event.type == "asr")
    assert asr_event.metadata["alternatives"] == list(alternatives)
    assert runtime.metadata_calls == [
        {
            "asr_rejection_reason": "",
            "asr_alternatives": list(alternatives),
        }
    ]


def test_voice_pipeline_exposes_asr_context_provenance_as_protocol_data() -> None:
    asr = FakeASR(
        "mở ghi chú attention",
        context_digest="a" * 64,
        context_provenance=("vault:7:wiki/attention.md:title", "session:thread:0:user"),
    )
    pipeline = VoicePipeline(
        asr=asr,
        llm=SpyLLM(),
        tts=SpyTTS(),
        assistant_runtime=FakeStreamingRuntime(["Mình đang mở ghi chú."]),
    )

    events = list(
        pipeline.turn_streaming(
            np.zeros(16000, dtype=np.float32),
            audio_sink=NullAudioPlayer(),
            min_sentence_chars=8,
        )
    )

    asr_event = next(event for event in events if event.type == "asr")
    assert asr_event.metadata["context_digest"] == "a" * 64
    assert asr_event.metadata["context_provenance"] == [
        "vault:7:wiki/attention.md:title",
        "session:thread:0:user",
    ]


def test_voice_pipeline_streaming_strips_markdown_before_tts():
    asr = FakeASR("tình hình")
    runtime = FakeStreamingRuntime(["**Tình hình:** nên ăn **đủ đạm**."])
    tts = SpyTTS()
    pipeline = VoicePipeline(asr=asr, llm=SpyLLM(), tts=tts, assistant_runtime=runtime)

    events = list(
        pipeline.turn_streaming(
            np.zeros(16000, dtype=np.float32),
            audio_sink=NullAudioPlayer(),
            min_sentence_chars=8,
        )
    )

    assert any(event.type == "sentence" and "**Tình hình:**" in event.text for event in events)
    assert any(
        event.type == "tts" and event.text == "Tình hình: nên ăn đủ đạm." for event in events
    )
    assert tts.calls == ["Tình hình: nên ăn đủ đạm."]


def test_voice_pipeline_keeps_citations_as_metadata_not_spoken_text() -> None:
    citation = KnowledgeCitation(
        path="wiki/learning/attention.md",
        title="Attention",
        line_start=12,
        line_end=18,
    )
    runtime = FakeStreamingRuntime(
        ["Attention dùng query, key và value [K1]."],
        citations=(citation,),
    )
    tts = SpyTTS()
    pipeline = VoicePipeline(
        asr=FakeASR("attention"),
        llm=SpyLLM(),
        tts=tts,
        assistant_runtime=runtime,
    )

    events = list(
        pipeline.turn_streaming(
            np.zeros(16000, dtype=np.float32),
            audio_sink=NullAudioPlayer(),
            min_sentence_chars=8,
        )
    )

    done = events[-1]
    assert done.type == "done"
    assert done.text == "Attention dùng query, key và value."
    runtime_event = next(event for event in events if event.type == "runtime")
    assert runtime_event.text == "Attention dùng query, key và value."
    sentence_event = next(event for event in events if event.type == "sentence")
    assert "[K1]" not in sentence_event.text
    assert done.metadata["citations"] == [
        {
            "label": "K1",
            "path": "wiki/learning/attention.md",
            "title": "Attention",
            "line_start": 12,
            "line_end": 18,
            "source": "knowledge",
        }
    ]
    assert tts.calls == ["Attention dùng query, key và value."]


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

    event_types = [event.type for event in events]
    assert event_types[:2] == ["asr", "repair"]
    assert event_types[-1] == "done"
    assert "sentence" in event_types
    assert "tts" in event_types
    assert "playback_started" in event_types
    assert "audio" in event_types
    assert event_types.index("sentence") < event_types.index("tts")
    repair = next(event for event in events if event.type == "repair")
    assert repair.metadata["repair_kind"] == "no_input"
    assert events[-1].metadata["rejected"] is True
    assert events[-1].metadata["terminal_status"] == "needs_clarification"
    assert all(
        event.metadata["delivery"] == "repair"
        for event in events
        if event.type in {"sentence", "tts"}
    )
    assert events[-1].metadata["repair_kind"] == "no_input"
    assert llm.calls == []
    repair_sentences = [
        event.text
        for event in events
        if event.type == "sentence" and event.metadata["delivery"] == "repair"
    ]
    assert len(tts.calls) == len(repair_sentences)
    assert len(tts.calls) >= 1
    assert tts.calls[0]


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
    runtime = FakeStreamingRuntime(["Xin chào bạn."])
    tts = FailingTTS()
    pipeline = VoicePipeline(asr=asr, llm=SpyLLM(), tts=tts, assistant_runtime=runtime)

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


def test_voice_pipeline_streaming_interrupt_event_stops_turn():
    """Barge-in mechanism (P0.1): a set interrupt_event ends the turn without
    speaking, emits an ``interrupted`` event, and flags it on ``done``.

    The event is pre-set so the assertions are deterministic (no thread race):
    the streaming loop breaks on its first iteration, so no sentence is ever
    submitted to the TTS pump. Mid-stream interruption is timing-dependent and
    is verified manually with a microphone (guide step 7-C).
    """
    asr = FakeASR("xin chao")
    runtime = FakeStreamingRuntime(["Câu một dài.", "Câu hai dài.", "Câu ba dài."])
    tts = SpyTTS()
    pipeline = VoicePipeline(asr=asr, llm=SpyLLM(), tts=tts, assistant_runtime=runtime)

    interrupt_event = threading.Event()
    interrupt_event.set()  # barge-in fires before the turn does any work

    events = list(
        pipeline.turn_streaming(
            np.zeros(16000, dtype=np.float32),
            audio_sink=NullAudioPlayer(),
            min_sentence_chars=8,
            interrupt_event=interrupt_event,
        )
    )

    event_types = [event.type for event in events]

    assert event_types == ["asr", "interrupted", "done"]
    assert events[-1].metadata["interrupted"] is True
    assert events[-1].metadata["terminal_status"] == "cancelled"
    # Interrupt fired first -> pump synthesizes and plays nothing.
    assert tts.calls == []
    assert "tts" not in event_types
    assert "audio" not in event_types


def test_voice_pipeline_streaming_without_interrupt_speaks_all():
    """Control for the interrupt test: with no interrupt, the turn runs fully."""
    asr = FakeASR("xin chao")
    runtime = FakeStreamingRuntime(["Câu một dài.", "Câu hai dài."])
    tts = SpyTTS()
    pipeline = VoicePipeline(asr=asr, llm=SpyLLM(), tts=tts, assistant_runtime=runtime)

    events = list(
        pipeline.turn_streaming(
            np.zeros(16000, dtype=np.float32),
            audio_sink=NullAudioPlayer(),
            min_sentence_chars=8,
            interrupt_event=threading.Event(),  # never set
        )
    )

    event_types = [event.type for event in events]

    assert "interrupted" not in event_types
    assert events[-1].metadata["interrupted"] is False
    assert tts.calls == ["Câu một dài.", "Câu hai dài."]
