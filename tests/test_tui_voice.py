from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Event

import numpy as np
import pytest

pytest.importorskip("textual")

from textual.widgets import Input

from soca.app.tui import SoCaTuiApp, TuiConfig
from soca.app.tui.voice import VoiceMonitorController
from soca.core import LLMUsage, ResolvedVoiceRuntimeConfig, StreamingEvent, VoiceRuntimeBundle
from soca.memory import SessionMemory
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


def test_voice_tui_listen_streams_pipeline_events_and_tracks_usage() -> None:
    config = make_config()
    pipeline = FakePipeline(
        [
            StreamingEvent(type="asr", text="xin chào"),
            StreamingEvent(type="llm_token", text="Xin "),
            StreamingEvent(type="llm_token", text="chào."),
            StreamingEvent(
                type="runtime",
                text="Xin chào.",
                metadata={
                    "route": "free_chat",
                    "llm_usage": LLMUsage(
                        prompt_tokens=11,
                        completion_tokens=4,
                        ttft_ms=3.0,
                        total_latency_ms=20.0,
                        tokens_per_second=80.0,
                    ),
                },
            ),
            StreamingEvent(
                type="tts",
                text="Xin chào.",
                latency_ms=30.0,
                metadata={"chunk_index": 0, "ttfa_ms": 50.0, "tts_latency_ms": 30.0},
            ),
            StreamingEvent(type="audio", text="Xin chào.", metadata={"playback_latency_ms": 4.0}),
            StreamingEvent(
                type="done",
                text="Xin chào.",
                latency_ms=100.0,
                metadata={
                    "rejected": False,
                    "runtime_route": "free_chat",
                    "runtime_blocked": False,
                    "stage_latencies_ms": {"asr": 10.0, "llm": 20.0},
                },
            ),
        ]
    )
    recorded_audio = np.zeros(1600, dtype=np.float32)

    def runtime_builder(_config: ResolvedVoiceRuntimeConfig) -> VoiceRuntimeBundle:
        return make_bundle(config, pipeline)

    async def run() -> None:
        app = SoCaTuiApp(
            TuiConfig(
                mode="voice",
                voice_runtime=config,
                no_model=False,
                show_splash=False,
                warmup_voice=False,
                auto_start_voice=False,
                voice_loop_max_turns=1,
            ),
            voice_runtime_builder=runtime_builder,
            voice_recorder=lambda *_args, **_kwargs: recorded_audio,
            voice_player=FakeAudioSink(),  # type: ignore[arg-type]
        )
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause()
            composer = app.query_one("#composer", Input)
            composer.value = "/listen"
            await pilot.press("enter")
            await pilot.pause()

            assert pipeline.audio_inputs == [recorded_audio]
            assert app.state.session_usage.total_turns == 1
            assert app.state.session_usage.total_prompt_tokens == 11
            assert app.state.last_turn_usage is not None
            assert app.state.last_turn_usage.ttfa_ms == 50.0
            assert app._voice_transcript == "xin chào"
            # Conversation lands in the timeline (chat-like), not a separate panel.
            timeline_text = app.timeline.plain_text
            assert "xin chào" in timeline_text
            assert "Xin chào." in timeline_text
            # Live status snapshot returns to idle after the turn completes.
            assert app._voice_turn.state == "idle"

    asyncio.run(run())


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


def test_repair_event_shows_followup_and_handover_switches_to_chat() -> None:
    from soca.app.tui.voice import VoiceMonitorEvent

    async def run() -> None:
        app = SoCaTuiApp(
            TuiConfig(mode="voice", no_model=True, show_splash=False, auto_start_voice=False),
        )
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause()
            # A handover repair (attempt 3) followed by done must switch to chat.
            app._render_voice_event(
                VoiceMonitorEvent(
                    "repair",
                    "Voice trục trặc, mình chuyển qua chat nhé.",
                    metadata={
                        "repair_kind": "no_input",
                        "repair_action": "handover_to_chat",
                        "repair_attempt": 3,
                        "handover_target": "chat",
                        "technical_reason": "no_speech",
                    },
                )
            )
            app._render_voice_event(
                VoiceMonitorEvent("done", "", metadata={"rejected": True})
            )
            await pilot.pause()

            assert "Follow-up:" in app.timeline.plain_text
            assert app.state.mode == "chat"

    asyncio.run(run())


def test_voice_mode_shows_status_bar_and_keeps_chat_body() -> None:
    async def run() -> None:
        app = SoCaTuiApp(
            TuiConfig(
                mode="voice",
                no_model=True,
                show_splash=False,
                warmup_voice=False,
                auto_start_voice=False,
            ),
        )
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause()
            # Voice mode: live status strip is visible; the chat body stays visible.
            assert app.voice_status.display is True
            assert app.query_one("#body").display is True

            composer = app.query_one("#composer", Input)
            composer.value = "/chat"
            await pilot.press("enter")
            await pilot.pause()

            # Chat mode hides the voice status strip; body still visible.
            assert app.voice_status.display is False
            assert app.query_one("#body").display is True

    asyncio.run(run())


def test_voice_tui_auto_starts_realtime_loop_in_voice_mode() -> None:
    config = make_config()
    pipeline = FakePipeline(
        [
            StreamingEvent(type="asr", text="xin chào"),
            StreamingEvent(type="runtime", text="Xin chào.", metadata={"route": "free_chat"}),
            StreamingEvent(type="done", text="Xin chào.", latency_ms=42.0),
        ]
    )
    recorded_audio = np.zeros(1600, dtype=np.float32)

    def runtime_builder(_config: ResolvedVoiceRuntimeConfig) -> VoiceRuntimeBundle:
        return make_bundle(config, pipeline)

    async def run() -> None:
        app = SoCaTuiApp(
            TuiConfig(
                mode="voice",
                voice_runtime=config,
                no_model=False,
                show_splash=False,
                warmup_voice=False,
                auto_start_voice=True,
                voice_loop_max_turns=1,
            ),
            voice_runtime_builder=runtime_builder,
            voice_recorder=lambda *_args, **_kwargs: recorded_audio,
            voice_player=FakeAudioSink(),  # type: ignore[arg-type]
        )
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause()
            await pilot.pause()

            assert pipeline.audio_inputs == [recorded_audio]
            assert app.state.session_usage.total_turns == 1
            assert app._voice_transcript == "xin chào"

    asyncio.run(run())


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

    events = []
    while True:
        item = queue.get_nowait()
        if item is None:
            break
        events.append(item)

    error = [event for event in events if event.type == "error"][0]
    assert "bad value(s) in fds_to_keep" in error.text
    assert "--extra tts-omnivoice" in error.text
    assert "Traceback" in error.metadata["traceback"]


def test_tui_voice_runtime_receives_shared_session_memory() -> None:
    config = make_config()
    captured: dict[str, SessionMemory | None] = {}
    pipeline = FakePipeline([StreamingEvent(type="done", text="", latency_ms=1.0)])

    def runtime_builder(
        _config: ResolvedVoiceRuntimeConfig,
        *,
        session_memory: SessionMemory | None = None,
    ) -> VoiceRuntimeBundle:
        captured["session"] = session_memory
        return make_bundle(config, pipeline)

    async def run() -> None:
        app = SoCaTuiApp(
            TuiConfig(
                mode="voice",
                voice_runtime=config,
                no_model=False,
                show_splash=False,
                warmup_voice=False,
                auto_start_voice=False,
                voice_loop_max_turns=1,
            ),
            voice_runtime_builder=runtime_builder,
            voice_recorder=lambda *_args, **_kwargs: np.zeros(1600, dtype=np.float32),
            voice_player=FakeAudioSink(),  # type: ignore[arg-type]
        )
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause()
            await app._start_voice_loop()
            await pilot.pause()

            assert captured["session"] is app.shared_session_memory

    asyncio.run(run())
