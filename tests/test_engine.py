from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from soca.app.engine import run_engine
from soca.app.text_runtime import TextRuntimeBundle, TextRuntimeConfig
from soca.core import ResolvedVoiceRuntimeConfig, StreamingEvent, VoiceRuntimeBundle
from soca.core.turn import RuntimeResult, RuntimeRoute


class ProtocolCapture:
    """Collects protocol lines and lets tests wait for a substring to appear."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self._lock = threading.Lock()
        self._seen = threading.Condition(self._lock)

    def write(self, text: str) -> None:
        with self._seen:
            self.lines.extend(part for part in text.splitlines() if part.strip())
            self._seen.notify_all()

    def flush(self) -> None:
        return None

    def wait_for(self, substring: str, timeout: float = 10.0) -> None:
        with self._seen:
            ok = self._seen.wait_for(
                lambda: any(substring in line for line in self.lines), timeout=timeout
            )
        assert ok, f"protocol never emitted {substring!r}; got: {self.lines}"

    def events(self) -> list[dict]:
        with self._lock:
            return [json.loads(line) for line in self.lines]


def _commands(capture: ProtocolCapture, first: dict | None, wait_substring: str | None):
    """stdin generator: send one command, wait for its effect, then quit."""
    if first is not None:
        yield json.dumps(first) + "\n"
        if wait_substring is not None:
            capture.wait_for(wait_substring)
    yield '{"cmd": "quit"}\n'


def make_text_config() -> TextRuntimeConfig:
    return TextRuntimeConfig(no_memory=True, vault=Path("/tmp/soca-test-vault"))


def make_voice_config() -> ResolvedVoiceRuntimeConfig:
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


def test_engine_hello_then_quit_emits_bye() -> None:
    capture = ProtocolCapture()
    code = run_engine(
        voice_config=None,
        text_config=make_text_config(),
        profile="baseline",
        no_model=True,
        stdin=_commands(capture, None, None),
        stdout=capture,
    )

    assert code == 0
    events = capture.events()
    assert events[0]["event"] == "hello"
    assert events[0]["version"] == 1
    assert events[-1]["event"] == "bye"


def test_engine_reports_invalid_json_without_crashing() -> None:
    capture = ProtocolCapture()

    def stdin():
        yield "this is not json\n"
        yield '{"cmd": "quit"}\n'

    code = run_engine(
        voice_config=None,
        text_config=make_text_config(),
        profile="baseline",
        no_model=True,
        stdin=stdin(),
        stdout=capture,
    )

    assert code == 0
    kinds = [event["event"] for event in capture.events()]
    assert "engine_error" in kinds


class _FakeAssistantRuntime:
    def run_text_turn(self, text: str, *, source: str, metadata: dict) -> RuntimeResult:
        return RuntimeResult(response_text=f"echo: {text}", route=RuntimeRoute.FREE_CHAT)


def _fake_text_builder(config: TextRuntimeConfig, session_memory=None) -> TextRuntimeBundle:
    return TextRuntimeBundle(
        runtime=_FakeAssistantRuntime(),  # type: ignore[arg-type]
        session_memory=session_memory,
        llm_status="fake",
        knowledge_status="fake",
        memory_status="fake",
    )


def test_engine_chat_roundtrip_emits_done_with_response() -> None:
    capture = ProtocolCapture()
    code = run_engine(
        voice_config=None,
        text_config=make_text_config(),
        profile="baseline",
        stdin=_commands(capture, {"cmd": "chat", "text": "xin chào"}, '"done"'),
        stdout=capture,
        text_runtime_builder=_fake_text_builder,
    )

    assert code == 0
    done = [e for e in capture.events() if e["event"] == "chat" and e["type"] == "done"]
    assert len(done) == 1
    assert done[0]["text"] == "echo: xin chào"
    assert done[0]["route"] == "free_chat"


@dataclass
class _FakeASR:
    boh_status: str = "disabled:test"
    confidence_guard_status: str = "disabled:test"


class _FakePipeline:
    def __init__(self, events: list[StreamingEvent]) -> None:
        self.events = events

    def turn_streaming(self, audio: np.ndarray, audio_sink):
        yield from self.events


class _FakeAudioSink:
    def play(self, audio: np.ndarray, sample_rate: int, blocking: bool = True):
        return None

    def stop(self) -> None:
        return None


class _FakeDetector:
    def speech_timestamps(self, audio: np.ndarray) -> list[dict[str, int]]:
        return [{"start": 0, "end": int(len(audio))}]


def _fake_voice_builder(config: ResolvedVoiceRuntimeConfig) -> VoiceRuntimeBundle:
    pipeline = _FakePipeline(
        [
            StreamingEvent(type="asr", text="xin chào"),
            StreamingEvent(
                type="done",
                text="Xin chào.",
                latency_ms=5.0,
                metadata={"rejected": False},
            ),
        ]
    )
    return VoiceRuntimeBundle(
        config=config,
        detector=_FakeDetector(),
        asr=_FakeASR(),  # type: ignore[arg-type]
        llm=object(),  # type: ignore[arg-type]
        tts=object(),
        assistant_runtime=object(),  # type: ignore[arg-type]
        pipeline=pipeline,  # type: ignore[arg-type]
        memory_status="disabled:test",
        knowledge_status="enabled:test",
    )


def _fake_recorder(detector, **kwargs) -> np.ndarray:
    return np.ones(1600, dtype=np.float32)


def test_engine_voice_start_streams_loop_events() -> None:
    capture = ProtocolCapture()
    code = run_engine(
        voice_config=make_voice_config(),
        text_config=make_text_config(),
        profile="baseline",
        stdin=_commands(capture, {"cmd": "voice_start", "max_turns": 1}, "loop_stopped"),
        stdout=capture,
        voice_runtime_builder=_fake_voice_builder,
        voice_recorder=_fake_recorder,
        voice_player=_FakeAudioSink(),
        warmup_voice=False,
    )

    assert code == 0
    voice_types = [e["type"] for e in capture.events() if e["event"] == "voice"]
    assert "loop_started" in voice_types
    assert "asr" in voice_types
    assert "loop_stopped" in voice_types
    asr = next(e for e in capture.events() if e["event"] == "voice" and e["type"] == "asr")
    assert asr["text"] == "xin chào"


def test_engine_no_model_rejects_voice_and_chat() -> None:
    capture = ProtocolCapture()

    def stdin():
        yield '{"cmd": "voice_start"}\n'
        yield '{"cmd": "chat", "text": "hi"}\n'
        yield '{"cmd": "quit"}\n'

    code = run_engine(
        voice_config=make_voice_config(),
        text_config=make_text_config(),
        profile="baseline",
        no_model=True,
        stdin=stdin(),
        stdout=capture,
    )

    assert code == 0
    errors = [e for e in capture.events() if e["event"] == "engine_error"]
    assert len(errors) == 2
    assert all("no-model" in e["message"] for e in errors)


def test_engine_memory_and_usage_commands() -> None:
    capture = ProtocolCapture()

    def stdin():
        yield json.dumps({"cmd": "chat", "text": "xin chào"}) + "\n"
        capture.wait_for('"done"')
        yield '{"cmd": "memory"}\n'
        capture.wait_for('"memory"')
        yield '{"cmd": "usage"}\n'
        capture.wait_for('"usage"')
        yield '{"cmd": "quit"}\n'

    code = run_engine(
        voice_config=None,
        text_config=TextRuntimeConfig(no_memory=False, vault=Path("/tmp/soca-test-vault")),
        profile="baseline",
        stdin=stdin(),
        stdout=capture,
        text_runtime_builder=_fake_text_builder,
    )

    assert code == 0
    events = capture.events()
    memory = next(e for e in events if e["event"] == "memory")
    assert memory["enabled"] is True
    usage = next(e for e in events if e["event"] == "usage")
    assert usage["turns"] == 1
