"""Tests for VoiceMonitorController — the thread-side voice loop adapter.

No UI framework involved: the controller streams VoiceMonitorEvents into a
queue consumed by the Ink UI (via `soca engine`) or any other front-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Event
from types import SimpleNamespace

import numpy as np

from soca.app.voice_controller import VoiceMonitorController
from soca.asr.selection import ASRSelection
from soca.core import ResolvedVoiceRuntimeConfig, StreamingEvent, VoiceRuntimeBundle
from soca.tts import TTSResult


@dataclass
class FakeASR:
    confidence_guard_status: str = "disabled:test"
    close_calls: int = 0

    def close(self) -> None:
        self.close_calls += 1


class FakePipeline:
    def __init__(self, events: list[StreamingEvent]) -> None:
        self.events = events
        self.audio_inputs: list[np.ndarray] = []
        self.audio_sink = None

    def turn_streaming(self, audio: np.ndarray, audio_sink):
        self.audio_inputs.append(audio)
        self.audio_sink = audio_sink
        yield from self.events


class FakeAudioSink:
    def __init__(self) -> None:
        self.play_calls = 0

    def play(self, audio: np.ndarray, sample_rate: int, blocking: bool = True):
        self.play_calls += 1
        return None

    def stop(self) -> None:
        return None


class FakeDetector:
    def __init__(self, *, has_speech: bool) -> None:
        self.has_speech = has_speech

    def speech_timestamps(self, audio: np.ndarray) -> list[dict[str, int]]:
        if not self.has_speech:
            return []
        return [{"start": 0, "end": int(len(audio))}]


class FakeTTS:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(self, text: str, voice: str | None = None) -> TTSResult:
        self.calls.append(text)
        return TTSResult(
            text=text,
            audio=np.zeros(160, dtype=np.float32),
            sample_rate=16_000,
            latency_ms=1.0,
            audio_duration_ms=10.0,
            rtf=0.1,
            voice=voice or "fake",
            engine="fake",
        )

    def list_voices(self) -> list[str]:
        return ["fake"]


def make_config() -> ResolvedVoiceRuntimeConfig:
    return ResolvedVoiceRuntimeConfig(
        profile_key="baseline",
        asr=ASRSelection.phowhisper("phowhisper_base"),
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


def make_bundle(
    config: ResolvedVoiceRuntimeConfig,
    pipeline: FakePipeline,
    *,
    detector: object | None = None,
    tts: object | None = None,
    llm: object | None = None,
) -> VoiceRuntimeBundle:
    return VoiceRuntimeBundle(
        config=config,
        detector=detector or object(),
        asr=FakeASR(),  # type: ignore[arg-type]
        llm=llm or object(),  # type: ignore[arg-type]
        tts=tts or object(),
        assistant_runtime=object(),  # type: ignore[arg-type]
        pipeline=pipeline,  # type: ignore[arg-type]
        memory_status="disabled:test",
        knowledge_status="enabled:test",
    )


def test_runtime_bundle_closes_llm_and_tts_handles() -> None:
    class CloseSpy:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    llm = CloseSpy()
    tts = CloseSpy()
    bundle = make_bundle(make_config(), FakePipeline([]), llm=llm, tts=tts)

    bundle.close()
    bundle.close()

    assert llm.close_calls == 1
    assert tts.close_calls == 1


def _drain_voice_events(queue: Queue) -> list:
    events = []
    while True:
        item = queue.get_nowait()
        if item is None:
            return events
        events.append(item)


def test_voice_monitor_passive_silence_skips_asr_pipeline() -> None:
    config = make_config()
    pipeline = FakePipeline([StreamingEvent(type="asr", text="should not run")])
    bundle = make_bundle(
        config,
        pipeline,
        detector=FakeDetector(has_speech=False),
        tts=FakeTTS(),
    )

    def runtime_builder(
        _config: ResolvedVoiceRuntimeConfig, *, session_memory=None
    ) -> VoiceRuntimeBundle:
        del session_memory
        return bundle

    queue: Queue = Queue()
    controller = VoiceMonitorController(
        config,
        runtime_builder=runtime_builder,
        recorder=lambda *_args, **_kwargs: np.zeros(1600, dtype=np.float32),
        player=FakeAudioSink(),  # type: ignore[arg-type]
        warmup=False,
    )
    controller.run_loop(queue, stop_event=Event(), max_turns=1)
    events = _drain_voice_events(queue)
    event_types = [event.type for event in events]

    # No speech -> the ASR/LLM pipeline is skipped entirely (efficiency)...
    assert pipeline.audio_inputs == []
    assert "asr" not in event_types


def test_voice_monitor_reports_microphone_level_for_nonempty_audio() -> None:
    config = make_config()
    pipeline = FakePipeline([StreamingEvent(type="asr", text="hello")])
    bundle = make_bundle(config, pipeline, detector=FakeDetector(has_speech=True))

    def runtime_builder(
        _config: ResolvedVoiceRuntimeConfig, *, session_memory=None
    ) -> VoiceRuntimeBundle:
        del session_memory
        return bundle

    queue: Queue = Queue()
    controller = VoiceMonitorController(
        config,
        runtime_builder=runtime_builder,
        recorder=lambda *_args, **_kwargs: np.full(1600, 0.25, dtype=np.float32),
        player=FakeAudioSink(),  # type: ignore[arg-type]
        warmup=False,
    )
    controller.run_loop(queue, stop_event=Event(), max_turns=1)

    events = _drain_voice_events(queue)
    level = next(event for event in events if event.type == "voice_level")
    assert level.metadata["source"] == "microphone"
    assert level.metadata["rms"] == 0.25


def test_voice_monitor_passive_silence_speaks_playful_call_out() -> None:
    config = make_config()
    pipeline = FakePipeline([StreamingEvent(type="asr", text="should not run")])
    fake_tts = FakeTTS()
    fake_player = FakeAudioSink()
    bundle = make_bundle(
        config,
        pipeline,
        detector=FakeDetector(has_speech=False),
        tts=fake_tts,
    )

    def runtime_builder(
        _config: ResolvedVoiceRuntimeConfig, *, session_memory=None
    ) -> VoiceRuntimeBundle:
        del session_memory
        return bundle

    queue: Queue = Queue()
    controller = VoiceMonitorController(
        config,
        runtime_builder=runtime_builder,
        recorder=lambda *_args, **_kwargs: np.zeros(1600, dtype=np.float32),
        player=fake_player,  # type: ignore[arg-type]
        warmup=False,
    )

    controller.run_loop(queue, stop_event=Event(), max_turns=1)
    events = _drain_voice_events(queue)
    event_types = [event.type for event in events]
    repair = next(event for event in events if event.type == "repair")

    # ...but SoCa still calls out the playful no_input follow-up, spoken in-worker.
    assert pipeline.audio_inputs == []
    assert fake_tts.calls == [repair.text]
    assert fake_player.play_calls == 1
    assert event_types.count("tts") == 1
    assert event_types.count("playback_started") == 1
    assert event_types.count("audio") == 1
    playback_started = next(
        event for event in events if event.type == "playback_started"
    )
    assert playback_started.metadata["audio_duration_ms"] == 10.0
    assert playback_started.metadata["sync_granularity"] == "audio_chunk"
    assert repair.metadata["repair_kind"] == "no_input"
    assert repair.metadata["repair_action"] == "reprompt"


def test_voice_monitor_reports_runtime_error_and_traceback() -> None:
    config = ResolvedVoiceRuntimeConfig(
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

    def failing_builder(
        _config: ResolvedVoiceRuntimeConfig, *, session_memory=None
    ) -> VoiceRuntimeBundle:
        del session_memory
        raise RuntimeError("Valtec runtime failed to load weights")

    queue: Queue = Queue()
    controller = VoiceMonitorController(config, runtime_builder=failing_builder, warmup=False)
    controller.run_loop(queue, stop_event=Event(), max_turns=1)

    events = _drain_voice_events(queue)
    error = [event for event in events if event.type == "error"][0]
    assert "Valtec runtime failed to load weights" in error.text
    assert "Traceback" in error.metadata["traceback"]


def test_voice_stop_cancels_active_llm_before_stopping_audio() -> None:
    calls: list[str] = []

    class CancelableLLM:
        def cancel(self) -> None:
            calls.append("llm")

    class OrderedAudioSink(FakeAudioSink):
        def stop(self) -> None:
            calls.append("audio")

    config = make_config()
    bundle = make_bundle(config, FakePipeline([]), llm=CancelableLLM())
    controller = VoiceMonitorController(
        config,
        runtime_builder=lambda _config, *, session_memory=None: bundle,
        player=OrderedAudioSink(),  # type: ignore[arg-type]
        warmup=False,
    )
    controller.bundle = bundle

    controller.stop()

    assert calls == ["llm", "audio"]


def test_voice_loop_reuses_one_runtime_and_closes_it_once() -> None:
    config = make_config()
    pipeline = FakePipeline([StreamingEvent(type="asr", text="hello")])
    bundle = make_bundle(config, pipeline, detector=FakeDetector(has_speech=True))
    build_calls = 0

    def runtime_builder(_config, *, session_memory=None):
        nonlocal build_calls
        del session_memory
        build_calls += 1
        return bundle

    controller = VoiceMonitorController(
        config,
        runtime_builder=runtime_builder,
        recorder=lambda *_args, **_kwargs: np.ones(1600, dtype=np.float32),
        player=FakeAudioSink(),  # type: ignore[arg-type]
        warmup=False,
    )
    queue: Queue = Queue()
    controller.run_loop(queue, stop_event=Event(), max_turns=2)

    assert build_calls == 1
    assert bundle.closed is True
    assert bundle.asr.close_calls == 1


class FakeContextAwareASR:
    """Mirrors QwenASRBackend: transcribe() accepts a context kwarg."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def transcribe(self, audio: np.ndarray, context: str | None = None):
        self.calls.append({"audio_len": len(audio), "context": context})
        return SimpleNamespace(text="hypothesis")

    def transcribe_partial(self, audio: np.ndarray):
        return self.transcribe(audio, context="")

    def close(self) -> None:
        return None


def _bundle_with_raw_asr(config: ResolvedVoiceRuntimeConfig, inner_asr: object) -> VoiceRuntimeBundle:
    return VoiceRuntimeBundle(
        config=config,
        detector=object(),
        asr=inner_asr,  # type: ignore[arg-type]
        llm=object(),  # type: ignore[arg-type]
        tts=object(),
        assistant_runtime=object(),  # type: ignore[arg-type]
        pipeline=object(),  # type: ignore[arg-type]
        memory_status="disabled:test",
        knowledge_status="enabled:test",
    )


def test_build_partial_transcriber_passes_empty_context_for_a_context_aware_backend() -> None:
    """Partial must call with context="" and never leak the final context
    onto the caption, since
    partial uses the raw backend directly with no guard in front of it."""
    config = make_config()
    inner = FakeContextAwareASR()
    bundle = _bundle_with_raw_asr(config, inner)

    transcribe = VoiceMonitorController._build_partial_transcriber(bundle)
    assert transcribe is not None

    text = transcribe(np.zeros(160, dtype=np.float32))

    assert text == "hypothesis"
    assert inner.calls == [{"audio_len": 160, "context": ""}]
