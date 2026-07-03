"""Tests for VoiceMonitorController — the thread-side voice loop adapter.

No UI framework involved: the controller streams VoiceMonitorEvents into a
queue consumed by the Ink UI (via `soca engine`) or any other front-end.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Event

import numpy as np

from soca.app.voice_controller import VoiceMonitorController
from soca.core import ResolvedVoiceRuntimeConfig, StreamingEvent, VoiceRuntimeBundle
from soca.tts import TTSResult


@dataclass
class FakeASR:
    boh_status: str = "disabled:test"
    confidence_guard_status: str = "disabled:test"


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
        asr_model="phowhisper_base",
        llm_model="arcee_vylinh_3b_q4_k_m",
        tts_model="valtec_multispeaker",
        tts_voice="NF",
        endpoint_silence_ms=700,
        max_record_ms=10_000,
        max_tokens=160,
        temperature=0.2,
        top_p=0.95,
        vault=Path("/tmp/soca-test-vault"),
        no_memory=True,
    )


def make_bundle(
    config: ResolvedVoiceRuntimeConfig,
    pipeline: FakePipeline,
    *,
    detector: object | None = None,
    tts: object | None = None,
) -> VoiceRuntimeBundle:
    return VoiceRuntimeBundle(
        config=config,
        detector=detector or object(),
        asr=FakeASR(),  # type: ignore[arg-type]
        llm=object(),  # type: ignore[arg-type]
        tts=tts or object(),
        assistant_runtime=object(),  # type: ignore[arg-type]
        pipeline=pipeline,  # type: ignore[arg-type]
        memory_status="disabled:test",
        knowledge_status="enabled:test",
    )


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

    def runtime_builder(_config: ResolvedVoiceRuntimeConfig) -> VoiceRuntimeBundle:
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

    def runtime_builder(_config: ResolvedVoiceRuntimeConfig) -> VoiceRuntimeBundle:
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
    assert event_types.count("audio") == 1
    assert repair.metadata["repair_kind"] == "no_input"
    assert repair.metadata["repair_action"] == "reprompt"


def test_voice_monitor_reports_omnivoice_extra_hint_and_traceback() -> None:
    config = ResolvedVoiceRuntimeConfig(
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

    def failing_builder(_config: ResolvedVoiceRuntimeConfig) -> VoiceRuntimeBundle:
        raise RuntimeError("OmniVoice runtime could not load omnivoice: bad value(s) in fds_to_keep")

    queue: Queue = Queue()
    controller = VoiceMonitorController(config, runtime_builder=failing_builder, warmup=False)
    controller.run_loop(queue, stop_event=Event(), max_turns=1)

    events = _drain_voice_events(queue)
    error = [event for event in events if event.type == "error"][0]
    assert "bad value(s) in fds_to_keep" in error.text
    assert "--extra tts-omnivoice" in error.text
    assert "Traceback" in error.metadata["traceback"]
