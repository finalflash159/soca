from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from soca.core import (
    NullAudioPlayer,
    PlaybackResult,
    RuntimeResult,
    RuntimeRoute,
    RuntimeStreamEvent,
    RuntimeTrace,
    VoicePipeline,
)
from soca.core.audio_join import crossfade_pcm
from soca.tts import TTSResult


@dataclass(frozen=True)
class FakeASRResult:
    text: str
    rejection_reason: str = ""


class FakeASR:
    def __init__(self, text: str, rejection_reason: str = "") -> None:
        self.text = text
        self.rejection_reason = rejection_reason
        self.calls = 0

    def transcribe(self, audio: np.ndarray) -> FakeASRResult:
        self.calls += 1
        return FakeASRResult(text=self.text, rejection_reason=self.rejection_reason)


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


class SlowAudioSink:
    def play(
        self, audio: np.ndarray, sample_rate: int, blocking: bool = True, interrupt_event=None
    ) -> PlaybackResult:
        time.sleep(0.05)
        return PlaybackResult(
            played=True,
            sample_rate=sample_rate,
            audio_duration_ms=100.0,
            latency_ms=50.0,
        )

    def stop(self) -> None:
        return None


class FailingAudioSink:
    def play(
        self, audio: np.ndarray, sample_rate: int, blocking: bool = True, interrupt_event=None
    ) -> PlaybackResult:
        raise RuntimeError("playback failed")

    def stop(self) -> None:
        return None


class SpyStreamingRuntime:
    """Runtime that exposes stream_text_turn (the new streaming contract)."""

    def __init__(
        self,
        sentences: list[str],
        *,
        route: RuntimeRoute = RuntimeRoute.FREE_CHAT,
        blocked: bool = False,
        used_tool: bool = False,
        used_llm: bool = True,
    ) -> None:
        self.sentences = sentences
        self.route = route
        self.blocked = blocked
        self.used_tool = used_tool
        self.used_llm = used_llm
        self.calls: list[dict] = []

    def stream_text_turn(
        self,
        text: str,
        *,
        source: str = "text",
        metadata=None,
        min_sentence_chars: int = 24,
        first_sentence_min_chars: int | None = None,
        first_clause_enabled: bool = True,
        first_clause_min_chars: int = 12,
        first_clause_min_words: int = 2,
        first_clause_max_scan_chars: int = 80,
    ) -> Iterator[RuntimeStreamEvent]:
        self.calls.append(
            {
                "text": text,
                "source": source,
                "metadata": metadata or {},
                "min_sentence_chars": min_sentence_chars,
            }
        )
        for sentence in self.sentences:
            for token in sentence.split(" "):
                yield RuntimeStreamEvent(type="token", text=token + " ")
            yield RuntimeStreamEvent(type="sentence", text=sentence)

        trace = RuntimeTrace(
            route=self.route,
            used_tool=self.used_tool,
            used_llm=self.used_llm,
            blocked=self.blocked,
        )
        yield RuntimeStreamEvent(
            type="result",
            result=RuntimeResult(
                response_text=" ".join(self.sentences),
                route=self.route,
                blocked=self.blocked,
                trace=trace,
            ),
        )


def test_pipeline_prefers_stream_text_turn_and_emits_streaming_events() -> None:
    asr = FakeASR("xin chào")
    tts = SpyTTS()
    runtime = SpyStreamingRuntime(["Câu một đủ dài rồi.", "Câu hai cũng đủ dài."])
    pipeline = VoicePipeline(asr=asr, llm=object(), tts=tts, assistant_runtime=runtime)

    events = list(
        pipeline.turn_streaming(
            np.zeros(16000, dtype=np.float32),
            audio_sink=NullAudioPlayer(),
            min_sentence_chars=8,
        )
    )
    types = [e.type for e in events]

    assert types[0] == "asr"
    assert "llm_token" in types  # real token streaming, not a single blob
    assert "sentence" in types
    assert "tts" in types
    assert "audio" in types
    assert "runtime" in types
    assert types[-1] == "done"
    assert tts.calls == ["Câu một đủ dài rồi.", "Câu hai cũng đủ dài."]
    assert runtime.calls[0]["min_sentence_chars"] == 8
    assert runtime.calls[0]["source"] == "asr"
    assert events[-1].metadata["runtime_route"] == RuntimeRoute.FREE_CHAT.value
    assert events[-1].metadata["rejected"] is False


def test_pipeline_stream_first_tts_carries_ttfa_metric() -> None:
    runtime = SpyStreamingRuntime(["Câu một đủ dài rồi.", "Câu hai cũng đủ dài."])
    tts = SpyTTS()
    pipeline = VoicePipeline(
        asr=FakeASR("xin chào"), llm=object(), tts=tts, assistant_runtime=runtime
    )

    events = list(
        pipeline.turn_streaming(
            np.zeros(16000, dtype=np.float32),
            audio_sink=NullAudioPlayer(),
            min_sentence_chars=8,
        )
    )
    tts_events = [e for e in events if e.type == "tts"]

    assert len(tts_events) == 2
    assert "ttfa_ms" in tts_events[0].metadata
    assert "ttfa_ms" not in tts_events[1].metadata


def test_pipeline_stream_synthesizes_next_chunk_while_audio_is_playing() -> None:
    runtime = SpyStreamingRuntime(["Câu trả lời đầu đủ dài.", "Câu trả lời sau đủ dài."])
    pipeline = VoicePipeline(
        asr=FakeASR("xin chào"),
        llm=object(),
        tts=SpyTTS(),
        assistant_runtime=runtime,
    )

    events = list(
        pipeline.turn_streaming(
            np.zeros(16000, dtype=np.float32),
            audio_sink=SlowAudioSink(),
            min_sentence_chars=8,
        )
    )
    types = [e.type for e in events]
    tts_indexes = [i for i, e in enumerate(events) if e.type == "tts"]
    first_audio_index = types.index("audio")

    # Both chunks are synthesized before the first one finishes playing.
    assert len(tts_indexes) == 2
    assert tts_indexes[1] < first_audio_index


def test_pipeline_stream_blocked_runtime_marks_done_blocked() -> None:
    runtime = SpyStreamingRuntime(
        ["Mình không thể xử lý yêu cầu này một cách an toàn."],
        route=RuntimeRoute.BLOCKED,
        blocked=True,
    )
    tts = SpyTTS()
    pipeline = VoicePipeline(
        asr=FakeASR("đọc private/secrets.md"),
        llm=object(),
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

    assert tts.calls == ["Mình không thể xử lý yêu cầu này một cách an toàn."]
    assert events[-1].type == "done"
    assert events[-1].metadata["runtime_blocked"] is True
    assert events[-1].metadata["rejected"] is False


def test_pipeline_stream_yields_error_when_playback_fails() -> None:
    runtime = SpyStreamingRuntime(["Một câu trả lời đủ dài để phát loa."])
    pipeline = VoicePipeline(
        asr=FakeASR("xin chào"),
        llm=object(),
        tts=SpyTTS(),
        assistant_runtime=runtime,
    )

    events = list(
        pipeline.turn_streaming(
            np.zeros(16000, dtype=np.float32),
            audio_sink=FailingAudioSink(),
            min_sentence_chars=8,
        )
    )

    assert any(e.type == "error" and e.text == "playback failed" for e in events)
    assert events[-1].type == "done"


class RecordingPlaybackSession:
    sample_rate = 24_000

    def __init__(self, *, block_first_write: float = 0.0) -> None:
        self.block_first_write = block_first_write
        self.writes: list[np.ndarray] = []
        self.output_underflow_count = 0
        self.finished = False
        self.first_write = threading.Event()

    def write(self, audio, *, interrupt_event=None):
        del interrupt_event
        arr = np.asarray(audio, dtype=np.float32).copy()
        self.writes.append(arr)
        self.first_write.set()
        if len(self.writes) == 1 and self.block_first_write:
            time.sleep(self.block_first_write)
        return PlaybackResult(
            played=bool(arr.size),
            sample_rate=self.sample_rate,
            audio_duration_ms=len(arr) / self.sample_rate * 1000.0,
            latency_ms=0.0,
        )

    def finish(self, *, interrupt_event=None):
        del interrupt_event
        self.finished = True
        return PlaybackResult(False, self.sample_rate, 0.0, 0.0)


class RecordingStreamingSink:
    def __init__(self, session: RecordingPlaybackSession) -> None:
        self.session = session

    def begin_playback(self, sample_rate):
        assert sample_rate == 24_000
        return self.session

    def play(self, *args, **kwargs):
        raise AssertionError("streaming sink must use begin_playback")

    def stop(self):
        return None


class TwoWaveTTS:
    def __init__(self, waves, *, require_first_write=None) -> None:
        self.waves = waves
        self.calls = 0
        self.require_first_write = require_first_write

    def synthesize(self, text, voice=None):
        del voice
        index = self.calls
        self.calls += 1
        if index == 1 and self.require_first_write is not None:
            assert self.require_first_write.wait(timeout=1.0)
        audio = self.waves[index]
        return TTSResult(
            text=text,
            audio=audio,
            sample_rate=24_000,
            latency_ms=1.0,
            audio_duration_ms=len(audio) / 24_000 * 1000.0,
            rtf=0.01,
            voice="NF",
            engine="fake",
        )


def test_session_writes_first_chunk_before_second_synthesis_finishes() -> None:
    session = RecordingPlaybackSession()
    tts = TwoWaveTTS(
        [
            np.linspace(-0.2, 0.2, 2_400, dtype=np.float32),
            np.linspace(0.2, -0.2, 2_400, dtype=np.float32),
        ],
        require_first_write=session.first_write,
    )
    pipeline = VoicePipeline(
        asr=FakeASR("xin chào"),
        llm=object(),
        tts=tts,
        assistant_runtime=SpyStreamingRuntime(["Câu đầu tiên.", "Câu thứ hai."]),
    )

    events = list(
        pipeline.turn_streaming(
            np.zeros(16_000, dtype=np.float32),
            audio_sink=RecordingStreamingSink(session),
            min_sentence_chars=8,
        )
    )

    assert session.first_write.is_set()
    assert tts.calls == 2
    assert session.finished is True
    assert any(
        event.type == "audio" and "audible_ttfa_ms" in (event.metadata or {})
        for event in events
    )


def test_session_pcm_matches_offline_crossfade_when_second_chunk_catches_up() -> None:
    first = np.linspace(-0.4, 0.4, 2_400, dtype=np.float32)
    second = np.linspace(0.4, -0.4, 2_400, dtype=np.float32)
    session = RecordingPlaybackSession(block_first_write=0.03)
    pipeline = VoicePipeline(
        asr=FakeASR("xin chào"),
        llm=object(),
        tts=TwoWaveTTS([first, second]),
        assistant_runtime=SpyStreamingRuntime(["Câu đầu tiên.", "Câu thứ hai."]),
        pcm_crossfade_ms=12.0,
    )

    list(
        pipeline.turn_streaming(
            np.zeros(16_000, dtype=np.float32),
            audio_sink=RecordingStreamingSink(session),
            min_sentence_chars=8,
        )
    )

    streamed = np.concatenate(session.writes)
    expected = crossfade_pcm(first, second, sample_rate=24_000, fade_ms=12.0)
    np.testing.assert_allclose(streamed, expected, atol=1e-6)
