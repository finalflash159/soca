from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.console import Console

from soca.app.voice_loop import run_voice_loop
from soca.core import LLMUsage, ResolvedVoiceRuntimeConfig, StreamingEvent, VoiceRuntimeBundle
from soca.tts import TTSResult


@dataclass
class FakeASRForBundle:
    boh_status: str = "disabled:test"
    confidence_guard_status: str = "disabled:test"


class FakePipeline:
    def __init__(self, events: list[StreamingEvent]) -> None:
        self.events = events
        self.audio_sink = None
        self.audio_inputs: list[np.ndarray] = []

    def turn_streaming(self, audio: np.ndarray, audio_sink):
        self.audio_inputs.append(audio)
        self.audio_sink = audio_sink
        yield from self.events


class FakeTTS:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def synthesize(self, text: str, voice: str | None = None) -> TTSResult:
        self.calls.append(text)
        return TTSResult(
            text=text,
            audio=np.zeros(2400, dtype=np.float32),
            sample_rate=24000,
            latency_ms=1.0,
            audio_duration_ms=100.0,
            rtf=0.01,
            voice=voice or "fake",
            engine="fake",
        )


class FakeAudioSink:
    def __init__(self) -> None:
        self.play_calls: list[tuple[np.ndarray, int, bool]] = []

    def play(self, audio: np.ndarray, sample_rate: int, *, blocking: bool = True):
        self.play_calls.append((audio, sample_rate, blocking))


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


def make_bundle(config: ResolvedVoiceRuntimeConfig, pipeline: FakePipeline, tts: FakeTTS):
    return VoiceRuntimeBundle(
        config=config,
        detector=object(),
        asr=FakeASRForBundle(),
        llm=object(),
        tts=tts,
        assistant_runtime=object(),
        pipeline=pipeline,
        memory_status="disabled:test",
        knowledge_status="disabled:test",
    )


def test_run_voice_loop_renders_streaming_events_without_real_models() -> None:
    config = make_config()
    pipeline = FakePipeline(
        [
            StreamingEvent(type="asr", text="xin chào"),
            StreamingEvent(
                type="runtime",
                text="Xin chào, mình là SoCa.",
                metadata={"route": "free_chat"},
            ),
            StreamingEvent(type="tts", text="Xin chào.", metadata={"ttfa_ms": 123.0}),
            StreamingEvent(type="audio", text="Xin chào."),
            StreamingEvent(type="done", latency_ms=456.0),
        ]
    )
    tts = FakeTTS()
    player = FakeAudioSink()
    console = Console(record=True, width=120)
    recorded_audio = np.zeros(1600, dtype=np.float32)
    input_prompts: list[str] = []

    result = run_voice_loop(
        config,
        console=console,
        input_fn=lambda prompt: input_prompts.append(prompt) or "",
        runtime_builder=lambda _: make_bundle(config, pipeline, tts),
        recorder=lambda *_args, **_kwargs: recorded_audio,
        player=player,
        max_turns=1,
        warmup=False,
    )

    output = console.export_text()
    assert result == 0
    assert "SoCa ready" in output
    assert "Runtime: free_chat" in output
    assert "Speaking chunk" in output
    assert "Done in 456 ms" in output
    assert pipeline.audio_inputs == [recorded_audio]
    assert pipeline.audio_sink is player
    assert tts.calls == []
    assert player.play_calls == []
    assert input_prompts == []


def test_run_voice_loop_usage_flag_renders_voice_metrics() -> None:
    config = make_config()
    pipeline = FakePipeline(
        [
            StreamingEvent(type="asr", text="xin chào"),
            StreamingEvent(
                type="runtime",
                text="Nên ăn nhẹ.",
                metadata={
                    "route": "free_chat",
                    "llm_usage": LLMUsage(
                        prompt_tokens=120,
                        completion_tokens=40,
                        ttft_ms=78.0,
                        tokens_per_second=62.0,
                    ),
                },
            ),
            StreamingEvent(
                type="tts",
                text="Nên ăn nhẹ.",
                metadata={"chunk_index": 0, "ttfa_ms": 410.0, "tts_latency_ms": 444.0},
            ),
            StreamingEvent(type="audio", text="Nên ăn nhẹ."),
            StreamingEvent(
                type="done",
                latency_ms=1400.0,
                metadata={
                    "rejected": False,
                    "runtime_route": "free_chat",
                    "runtime_blocked": False,
                    "stage_latencies_ms": {"asr": 480.0, "llm": 530.0},
                },
            ),
        ]
    )
    console = Console(record=True, width=120)

    run_voice_loop(
        config,
        show_usage=True,
        console=console,
        input_fn=lambda _: "",
        runtime_builder=lambda _: make_bundle(config, pipeline, FakeTTS()),
        recorder=lambda *_args, **_kwargs: np.zeros(1600, dtype=np.float32),
        player=FakeAudioSink(),
        max_turns=1,
        warmup=False,
    )

    out = console.export_text()
    assert "route=free_chat" in out
    assert "ASR 480ms" in out
    assert "TTFA 410ms" in out
    assert "total 1400ms" in out


def test_run_voice_loop_streams_llm_tokens_live_and_skips_runtime_panel() -> None:
    config = make_config()
    pipeline = FakePipeline(
        [
            StreamingEvent(type="asr", text="xin chào"),
            StreamingEvent(type="llm_token", text="Xin "),
            StreamingEvent(type="llm_token", text="chào."),
            StreamingEvent(
                type="runtime",
                text="Xin chào.",
                metadata={"route": "free_chat"},
            ),
            StreamingEvent(type="tts", text="Xin chào.", metadata={"ttfa_ms": 12.0}),
            StreamingEvent(type="audio", text="Xin chào."),
            StreamingEvent(type="done", latency_ms=200.0),
        ]
    )
    console = Console(record=True, width=120)

    run_voice_loop(
        config,
        console=console,
        input_fn=lambda _: "",
        runtime_builder=lambda _: make_bundle(config, pipeline, FakeTTS()),
        recorder=lambda *_args, **_kwargs: np.zeros(1600, dtype=np.float32),
        player=FakeAudioSink(),
        max_turns=1,
        warmup=False,
    )

    output = console.export_text()
    # Tokens are rendered live under a speaker prefix.
    assert "SoCa:" in output
    assert "Xin chào." in output
    # The redundant runtime panel is suppressed once tokens streamed live.
    assert "Runtime: free_chat" not in output
    assert "Speaking chunk" in output


def test_run_voice_loop_can_wait_for_enter_in_manual_mode() -> None:
    config = make_config()
    pipeline = FakePipeline([StreamingEvent(type="done", latency_ms=1.0)])
    input_prompts: list[str] = []

    run_voice_loop(
        config,
        press_enter_to_record=True,
        console=Console(record=True),
        input_fn=lambda prompt: input_prompts.append(prompt) or "",
        runtime_builder=lambda _: make_bundle(config, pipeline, FakeTTS()),
        recorder=lambda *_args, **_kwargs: np.zeros(1600, dtype=np.float32),
        player=FakeAudioSink(),
        max_turns=1,
        warmup=False,
    )

    assert input_prompts == ["\nPress ENTER and speak. Ctrl+C to quit."]


def test_run_voice_loop_speaks_rejection_fallback_by_default() -> None:
    config = make_config()
    fallback = "Mình chưa nghe rõ, bạn nói lại nhé."
    pipeline = FakePipeline([StreamingEvent(type="done", text=fallback, metadata={"rejected": True})])
    tts = FakeTTS()
    player = FakeAudioSink()

    run_voice_loop(
        config,
        console=Console(record=True),
        input_fn=lambda _: "",
        runtime_builder=lambda _: make_bundle(config, pipeline, tts),
        recorder=lambda *_args, **_kwargs: np.zeros(1600, dtype=np.float32),
        player=player,
        max_turns=1,
        warmup=False,
    )

    assert tts.calls == [fallback]
    assert len(player.play_calls) == 1
    assert player.play_calls[0][1] == 24000


def test_run_voice_loop_does_not_double_speak_pipeline_rejection_audio() -> None:
    config = make_config()
    fallback = "Mình chưa nghe rõ, bạn nói lại nhé."
    pipeline = FakePipeline(
        [
            StreamingEvent(type="tts", text=fallback, metadata={"chunk_index": 0}),
            StreamingEvent(type="audio", text=fallback),
            StreamingEvent(type="done", text=fallback, metadata={"rejected": True}),
        ]
    )
    tts = FakeTTS()
    player = FakeAudioSink()

    run_voice_loop(
        config,
        console=Console(record=True),
        input_fn=lambda _: "",
        runtime_builder=lambda _: make_bundle(config, pipeline, tts),
        recorder=lambda *_args, **_kwargs: np.zeros(1600, dtype=np.float32),
        player=player,
        max_turns=1,
        warmup=False,
    )

    assert tts.calls == []
    assert player.play_calls == []


def test_run_voice_loop_can_suppress_rejection_fallback_speech() -> None:
    config = make_config()
    pipeline = FakePipeline(
        [
            StreamingEvent(
                type="done",
                text="Mình chưa nghe rõ, bạn nói lại nhé.",
                metadata={"rejected": True},
            )
        ]
    )
    tts = FakeTTS()
    player = FakeAudioSink()

    run_voice_loop(
        config,
        no_speak_rejections=True,
        console=Console(record=True),
        input_fn=lambda _: "",
        runtime_builder=lambda _: make_bundle(config, pipeline, tts),
        recorder=lambda *_args, **_kwargs: np.zeros(1600, dtype=np.float32),
        player=player,
        max_turns=1,
        warmup=False,
    )

    assert tts.calls == []
    assert player.play_calls == []
