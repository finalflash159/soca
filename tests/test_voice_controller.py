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
import pytest

from soca.app import voice_controller as voice_controller_module
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


def test_voice_controller_reports_effective_endpoint_overrides(monkeypatch):
    monkeypatch.setenv("SOCA_ENDPOINT_FLOOR_MS", "2100")
    monkeypatch.setenv("SOCA_ENDPOINT_CEIL_MS", "3600")
    config = make_config()
    bundle = make_bundle(config, FakePipeline([]))
    controller = VoiceMonitorController(config, warmup=False)

    effective = controller._endpoint_config(bundle)

    assert effective.floor_silence_ms == 2100
    assert effective.ceil_silence_ms == 3600


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


def test_voice_monitor_passive_silence_waits_before_calling_out() -> None:
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
    # A fresh silent session must not speak immediately. The first callout is
    # delayed by the configured five-minute interval.
    assert pipeline.audio_inputs == []
    assert fake_tts.calls == []
    assert fake_player.play_calls == 0
    assert "repair" not in event_types


def test_voice_monitor_passive_silence_waits_five_minutes_and_stops_after_three_callouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config()
    fake_tts = FakeTTS()
    fake_player = FakeAudioSink()
    bundle = make_bundle(
        config,
        FakePipeline([]),
        detector=FakeDetector(has_speech=False),
        tts=fake_tts,
    )
    controller = VoiceMonitorController(
        config,
        runtime_builder=lambda _config, *, session_memory=None: bundle,
        player=fake_player,  # type: ignore[arg-type]
        warmup=False,
    )
    now = [0.0]
    monkeypatch.setattr(voice_controller_module.time, "perf_counter", lambda: now[0])
    controller._idle_started_at = 0.0
    queue: Queue = Queue()
    stop_event = Event()

    controller._handle_passive_silence(bundle, queue, stop_event=stop_event)
    assert fake_tts.calls == []

    for minute in (5, 10, 15):
        now[0] = minute * 60.0
        controller._handle_passive_silence(bundle, queue, stop_event=stop_event)

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    repairs = [event for event in events if event.type == "repair"]
    assert len(repairs) == 3
    assert repairs[-1].metadata["shutdown_after_callout"] is True
    assert repairs[-1].metadata["callout_interval_ms"] == 300_000
    done_events = [event for event in events if event.type == "done"]
    assert done_events[-1].metadata["terminal_status"] == "cancelled"
    assert stop_event.is_set()
    assert controller._stop_reason == "passive_silence_callout_limit"

    now[0] = 20 * 60.0
    controller._handle_passive_silence(bundle, queue, stop_event=stop_event)
    assert queue.empty()


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


def test_voice_stop_keeps_teardown_observable_when_component_close_fails() -> None:
    config = make_config()
    bundle = make_bundle(config, FakePipeline([]))

    def fail_close() -> None:
        raise RuntimeError("native close failed")

    bundle.close = fail_close  # type: ignore[method-assign]
    controller = VoiceMonitorController(
        config,
        runtime_builder=lambda _config, *, session_memory=None: bundle,
        player=FakeAudioSink(),  # type: ignore[arg-type]
        warmup=False,
    )
    controller.bundle = bundle

    with pytest.raises(RuntimeError, match="native close failed"):
        controller.stop()

    assert controller.bundle is bundle


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
