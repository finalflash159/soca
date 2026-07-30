from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from soca.app.engine import _memory_protocol_mode, _retrieval_trace_payload, run_engine
from soca.app.text_runtime import TextRuntimeBundle, TextRuntimeConfig
from soca.core import ResolvedVoiceRuntimeConfig, StreamingEvent, VoiceRuntimeBundle
from soca.core.turn import RuntimeResult, RuntimeRoute, RuntimeTrace


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


def test_memory_protocol_mode_exposes_degraded_fallback() -> None:
    assert _memory_protocol_mode("blob", "retrieval_unavailable", 0) == "degraded"
    assert _memory_protocol_mode("retrieved", "", 0) == "retrieved"


def test_retrieval_trace_preserves_backend_scores_and_rejections() -> None:
    from types import SimpleNamespace

    hit = SimpleNamespace(
        document=SimpleNamespace(path="wiki/learning/bayes.md"),
        score=0.91,
        retrieval_backend="hybrid",
        sparse_score=0.7,
        dense_score=0.8,
        fusion_score=0.91,
    )
    decision = SimpleNamespace(
        source="knowledge",
        rejected_count=2,
        as_dict=lambda: {"status": "supported", "rejected_count": 2},
    )
    trace = SimpleNamespace(
        tool_router_tier="semantic",
        stage_latencies_ms={"knowledge": 4.5},
        evidence_decisions=(decision,),
    )
    result = SimpleNamespace(frame=SimpleNamespace(text="định lý Bayes"))

    payload = _retrieval_trace_payload(result, trace, (hit,))

    assert payload["columns"] == [
        {
            "source": "hybrid",
            "hits": [
                {
                    "path": "wiki/learning/bayes.md",
                    "score": 0.91,
                    "sparse_score": 0.7,
                    "dense_score": 0.8,
                    "fusion_score": 0.91,
                }
            ],
        }
    ]
    assert payload["rejected_count"] == 2


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
    assert events[0]["version"] == 2
    assert events[0]["supported_versions"] == [1, 2]
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


def test_engine_status_does_not_load_embedding_runtime(monkeypatch, tmp_path: Path) -> None:
    from soca.knowledge.indexing import models

    def forbidden_load(*args, **kwargs):
        raise AssertionError("status must not load the embedding runtime")

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(models, "load_model", forbidden_load)
    capture = ProtocolCapture()
    code = run_engine(
        voice_config=None,
        text_config=make_text_config(),
        profile="baseline",
        no_model=True,
        stdin=_commands(capture, {"cmd": "status"}, '"event": "status"'),
        stdout=capture,
    )

    assert code == 0
    event = next(item for item in capture.events() if item["event"] == "status")
    embedding = next(
        item for item in event["runtime_components"] if item["id"] == "embedding"
    )
    assert embedding["status"] == "missing"
    assert event["knowledge_index"]["dense_state"] == "model_missing"


def test_engine_uses_local_defaults_when_saved_llm_settings_are_invalid() -> None:
    capture = ProtocolCapture()

    def stdin():
        yield '{"cmd": "llm_config"}\n'
        yield '{"cmd": "quit"}\n'

    def invalid_settings_loader():
        raise ValueError("settings file is malformed")

    code = run_engine(
        voice_config=None,
        text_config=make_text_config(),
        profile="baseline",
        no_model=True,
        stdin=stdin(),
        stdout=capture,
        llm_settings_loader=invalid_settings_loader,
    )

    assert code == 0
    events = capture.events()
    assert events[0]["event"] == "hello"
    assert any(
        event["event"] == "engine_error" and "cấu hình llm" in event["message"].lower()
        for event in events
    )
    config = next(event for event in events if event["event"] == "llm_config")
    assert config["backend"] == "local"


class _FakeAssistantRuntime:
    def run_text_turn(self, text: str, *, source: str, metadata: dict) -> RuntimeResult:
        return RuntimeResult(response_text=f"echo: {text}", route=RuntimeRoute.FREE_CHAT)


class _ManifestAssistantRuntime:
    def run_text_turn(self, text: str, *, source: str, metadata: dict) -> RuntimeResult:
        trace = RuntimeTrace(
            route=RuntimeRoute.FREE_CHAT,
            prompt_manifest={
                "model_id": "test-model",
                "context_window": 2_048,
                "token_counter": "engine",
                "requested_output_tokens": 4_096,
                "effective_output_tokens": 1_600,
                "input_budget_tokens": 416,
                "prompt_tokens": 32,
                "prompt_hash": "abc123",
                "components": [
                    {
                        "component_id": "system",
                        "tokens": 20,
                        "included": True,
                        "required": True,
                        "priority": 0,
                    },
                    {
                        "component_id": "archive",
                        "tokens": 400,
                        "included": False,
                        "required": False,
                        "priority": 50,
                    },
                ],
            },
        )
        return RuntimeResult(
            response_text=f"echo: {text}",
            route=RuntimeRoute.FREE_CHAT,
            trace=trace,
        )


class _FailingAssistantRuntime:
    def run_text_turn(self, text: str, *, source: str, metadata: dict) -> RuntimeResult:
        raise RuntimeError("synthetic runtime failure")


def _fake_text_builder(config: TextRuntimeConfig, session_memory=None) -> TextRuntimeBundle:
    return TextRuntimeBundle(
        runtime=_FakeAssistantRuntime(),  # type: ignore[arg-type]
        session_memory=session_memory,
        llm_status="fake",
        knowledge_status="fake",
        memory_status="fake",
    )


def _manifest_text_builder(config: TextRuntimeConfig, session_memory=None) -> TextRuntimeBundle:
    return TextRuntimeBundle(
        runtime=_ManifestAssistantRuntime(),  # type: ignore[arg-type]
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
    progress = [e for e in capture.events() if e["event"] == "turn_progress"]
    assert [event["phase"] for event in progress] == [
        "preparing",
        "analyzing",
        "complete",
    ]
    assert progress[-1]["status"] == "done"
    assert all(progress_event["run_id"] == progress[0]["run_id"] for progress_event in progress)
    assert all(isinstance(progress_event["sequence"], int) for progress_event in progress)
    assert progress[-1]["terminal_status"] == "achieved"
    workflow = [
        event
        for event in capture.events()
        if event["event"] in {"turn_started", "answer_delta", "turn_terminal"}
    ]
    assert workflow[-1]["event"] == "turn_terminal"
    assert workflow[-1]["payload"]["terminal_status"] == "achieved"


def test_engine_context_exposes_last_prompt_manifest() -> None:
    capture = ProtocolCapture()
    code = run_engine(
        voice_config=None,
        text_config=make_text_config(),
        profile="baseline",
        stdin=_commands(capture, {"cmd": "chat", "text": "xin chào"}, '"done"'),
        stdout=capture,
        text_runtime_builder=_manifest_text_builder,
    )

    assert code == 0
    context = [event for event in capture.events() if event["event"] == "context"][-1]
    assert context["estimated"] is False
    assert context["prompt_hash"] == "abc123"
    assert context["model_context_tokens"] == 2_048
    assert context["output_reserve_tokens"] == 1_600
    archive = next(item for item in context["components"] if item["id"] == "archive")
    assert archive["included"] is False
    assert archive["policy"] == "on_demand"


def test_engine_chat_exception_does_not_emit_completed_progress() -> None:
    capture = ProtocolCapture()

    def failing_builder(config: TextRuntimeConfig, session_memory=None) -> TextRuntimeBundle:
        return TextRuntimeBundle(
            runtime=_FailingAssistantRuntime(),  # type: ignore[arg-type]
            session_memory=session_memory,
            llm_status="fake",
            knowledge_status="fake",
            memory_status="fake",
        )

    def stdin():
        yield '{"cmd": "chat", "text": "xin chào"}\n'
        capture.wait_for('"type": "error"')
        yield '{"cmd": "quit"}\n'

    code = run_engine(
        voice_config=None,
        text_config=make_text_config(),
        profile="baseline",
        stdin=stdin(),
        stdout=capture,
        text_runtime_builder=failing_builder,
    )

    assert code == 0
    progress = [event for event in capture.events() if event["event"] == "turn_progress"]
    assert not any(event["status"] == "done" for event in progress)
    assert progress[-1]["status"] == "failed"
    assert progress[-1]["terminal_status"] == "system_failure"


@dataclass
class _FakeASR:
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
                type="runtime",
                text="Theo memory [M1].",
                metadata={
                    "router_tier": "semantic",
                    "router_reason": "retrieval_request",
                    "router_disposition": "retrieval_request",
                    "router_handler": "memory",
                    "router_selected_routes": ["retrieval_request"],
                    "router_sources": ["memory"],
                    "router_scores": {"retrieval_request": 0.91},
                    "router_source_scores": {"memory": 0.93},
                    "router_runner_up": "unresolved",
                    "router_margin": 0.12,
                    "router_latency_ms": 4.2,
                    "evidence_status": "supported",
                    "answer_policy": "grounded",
                    "answer_policy_reason": "supported_evidence",
                    "grounding_policy_version": "grounding-v1",
                    "citation_count": 1,
                    "memory_access_plan": {
                        "include_core": True,
                        "include_working": True,
                        "archive_mode": "semantic",
                        "archive_query": "ghi chú đã lưu",
                        "reason": "semantic_source_selection",
                    },
                },
            ),
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
    router_trace = next(e for e in capture.events() if e["event"] == "router_trace")
    assert router_trace["evidence_status"] == "supported"
    assert router_trace["answer_policy"] == "grounded"
    assert router_trace["citation_count"] == 1
    assert router_trace["memory_access_plan"]["archive_mode"] == "semantic"


def test_engine_passes_one_selected_settings_and_goal_store_to_voice_builder(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_build_voice_runtime(config, **kwargs):
        captured.update(kwargs)
        return _fake_voice_builder(config)

    monkeypatch.setattr("soca.app.engine.build_voice_runtime", fake_build_voice_runtime)

    from soca.app.engine import SocaEngine, _ProtocolWriter

    instance = SocaEngine(
        voice_config=make_voice_config(),
        text_config=make_text_config(),
        profile="baseline",
        writer=_ProtocolWriter(ProtocolCapture()),
        voice_player=_FakeAudioSink(),
        warmup_voice=False,
    )
    controller = instance._ensure_voice_controller()
    controller._build_runtime_bundle()

    assert captured["llm_settings"] is instance.llm_settings
    assert captured["secret_store"] is instance.secret_store
    assert captured["active_goal_store"] is instance.active_goal_store


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
        yield '{"cmd": "context"}\n'
        capture.wait_for('"resident_prompt_tokens"')
        yield '{"cmd": "memory_compact", "action": "status"}\n'
        capture.wait_for('"memory_compaction"')
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
    assert memory["stats"]["hard_limit_tokens"] == 16_384
    contexts = [e for e in events if e["event"] == "context"]
    context = contexts[-1]
    assert context["estimated"] is True
    assert context["token_counter"] == "utf8_bytes_div_4"
    assert context["session"]["high_watermark_tokens"] == 15_000
    assert context["model_context_tokens"] == 32_768
    assert context["output_reserve_tokens"] == 4_096
    assert context["ready"] is True
    assert context["prompt_hash"]
    assert context["input_budget_tokens"] == 28_544
    assert context["available_dynamic_tokens"] <= context["input_budget_tokens"]
    assert {component["id"] for component in context["components"]} >= {
        "system",
        "core_memory",
        "working_summary",
        "recent_conversation",
        "knowledge",
        "archive_memory",
        "current_input",
    }
    knowledge = next(
        component for component in context["components"] if component["id"] == "knowledge"
    )
    assert knowledge["tokens"] is None
    assert knowledge["policy"] == "on_demand"
    compaction = next(e for e in events if e["event"] == "memory_compaction")
    assert compaction["status"] == "idle"
    usage = next(e for e in events if e["event"] == "usage")
    assert usage["turns"] == 1
