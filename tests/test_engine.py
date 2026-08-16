from __future__ import annotations

import dataclasses
import json
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from soca.app.engine import (
    SocaEngine,
    _memory_protocol_mode,
    _retrieval_trace_payload,
    run_engine,
)
from soca.app.text_runtime import TextRuntimeBundle, TextRuntimeConfig
from soca.asr.selection import ASRSelection
from soca.core import ResolvedVoiceRuntimeConfig, StreamingEvent, VoiceRuntimeBundle
from soca.core.turn import RuntimeResult, RuntimeRoute, RuntimeStreamEvent, RuntimeTrace
from soca.knowledge import KnowledgeCitation


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


def test_memory_protocol_mode_exposes_degraded_retrieval_state() -> None:
    assert _memory_protocol_mode("none", "retrieval_unavailable", 0) == "degraded"
    assert _memory_protocol_mode("retrieved", "", 0) == "retrieved"
    assert _memory_protocol_mode("unknown", "", 0) == "none"


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


def test_pending_proposal_telemetry_reports_unavailable_state(caplog) -> None:
    class BrokenCommands:
        def list_pending(self):
            raise OSError("proposal store unavailable")

    engine = object.__new__(SocaEngine)
    engine._memory_commands = lambda: BrokenCommands()

    with caplog.at_level("WARNING", logger="soca.app.engine"):
        count = engine._pending_proposal_count()

    assert count is None
    assert "proposal telemetry unavailable" in caplog.text


def test_dispose_text_bundle_closes_owner_before_detaching() -> None:
    class Bundle:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    engine = object.__new__(SocaEngine)
    bundle = Bundle()
    engine.text_bundle = bundle

    assert engine._dispose_text_bundle() is True
    assert bundle.close_calls == 1
    assert engine.text_bundle is None


def test_dispose_text_bundle_keeps_owner_after_cleanup_failure() -> None:
    class Bundle:
        def close(self) -> None:
            raise RuntimeError("watcher did not stop")

    class Writer:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def emit(self, event: dict) -> None:
            self.events.append(event)

    engine = object.__new__(SocaEngine)
    engine.text_bundle = Bundle()
    engine.writer = Writer()

    assert engine._dispose_text_bundle() is False
    assert engine.text_bundle is not None
    assert engine.writer.events == [
        {
            "event": "engine_error",
            "message": "text runtime cleanup failed",
            "code": "runtime_cleanup_failed",
            "detail": "RuntimeError",
        }
    ]


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
    assert events[0]["supported_versions"] == [2]
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
    qwen = next(
        item
        for item in event["runtime_components"]
        if item["id"] == "qwen_asr_release"
    )
    assert embedding["status"] == "missing"
    assert qwen["status"] in {"missing", "invalid", "provisioned", "unsupported"}
    assert "fallback" in qwen["detail"] or qwen["status"] != "unsupported"
    assert event["knowledge_index"]["dense_state"] == "model_missing"


def test_engine_blocks_runtime_when_saved_llm_settings_are_invalid() -> None:
    capture = ProtocolCapture()

    def stdin():
        yield '{"cmd": "llm_config"}\n'
        yield '{"cmd": "status"}\n'
        yield '{"cmd": "quit"}\n'

    def invalid_settings_loader():
        raise ValueError("settings file is malformed")

    code = run_engine(
        voice_config=None,
        text_config=make_text_config(),
        profile="baseline",
        no_model=False,
        stdin=stdin(),
        stdout=capture,
        llm_settings_loader=invalid_settings_loader,
    )

    assert code == 0
    events = capture.events()
    assert events[0]["event"] == "hello"
    assert any(
        event["event"] == "engine_error"
        and event.get("code") == "llm_settings_invalid"
        for event in events
    )
    config = next(event for event in events if event["event"] == "llm_config")
    assert config["backend"] == "local"
    assert config["runtime_ready"] is False
    assert config["settings_error"]
    status = next(event for event in events if event["event"] == "status")
    chat = next(
        item for item in status["runtime_components"] if item["id"] == "chat_llm"
    )
    assert chat["status"] == "invalid"


def test_voice_profile_selection_invalidates_before_persisting(monkeypatch) -> None:
    from types import SimpleNamespace

    from soca.app.engine import SocaEngine, _ProtocolWriter

    selected: list[str] = []
    fake_profile = SimpleNamespace(
        key="qwen-release",
        asr=ASRSelection.phowhisper("phowhisper_base"),
        asr_model="phowhisper_base",
    )
    monkeypatch.setattr(
        "soca.app.engine.get_voice_runtime_profile",
        lambda _key: fake_profile,
    )
    monkeypatch.setattr(
        "soca.app.profiles.asr_readiness",
        lambda _selection: SimpleNamespace(ok=True, status="ok", detail="ready"),
    )

    instance = SocaEngine(
        voice_config=make_voice_config(),
        text_config=make_text_config(),
        profile="baseline",
        writer=_ProtocolWriter(ProtocolCapture()),
        warmup_voice=False,
        voice_profile_saver=selected.append,
    )

    instance._cmd_voice_profile_select({"profile": "qwen-release"})

    assert selected == ["qwen-release"]
    assert instance.profile == "qwen-release"
    assert instance.voice_config is not None
    assert instance.voice_config.profile_key == "qwen-release"


def test_voice_profile_selection_does_not_persist_when_invalidation_fails(monkeypatch) -> None:
    from types import SimpleNamespace

    from soca.app.engine import SocaEngine, _ProtocolWriter

    selected: list[str] = []
    fake_profile = SimpleNamespace(
        key="qwen-release",
        asr=ASRSelection.phowhisper("phowhisper_base"),
        asr_model="phowhisper_base",
    )
    monkeypatch.setattr(
        "soca.app.engine.get_voice_runtime_profile",
        lambda _key: fake_profile,
    )
    monkeypatch.setattr(
        "soca.app.profiles.asr_readiness",
        lambda _selection: SimpleNamespace(ok=True, status="ok", detail="ready"),
    )

    instance = SocaEngine(
        voice_config=make_voice_config(),
        text_config=make_text_config(),
        profile="baseline",
        writer=_ProtocolWriter(ProtocolCapture()),
        warmup_voice=False,
        voice_profile_saver=selected.append,
    )
    monkeypatch.setattr(instance, "_invalidate_voice_runtime", lambda: False)

    instance._cmd_voice_profile_select({"profile": "qwen-release"})

    assert selected == []
    assert instance.profile == "baseline"
    assert instance.voice_config is not None
    assert instance.voice_config.profile_key == "baseline"


def test_voice_profile_selection_surfaces_oserror_without_applying(monkeypatch) -> None:
    from types import SimpleNamespace

    from soca.app.engine import SocaEngine, _ProtocolWriter

    fake_profile = SimpleNamespace(
        key="qwen-release",
        asr=ASRSelection.phowhisper("phowhisper_base"),
        asr_model="phowhisper_base",
    )
    monkeypatch.setattr(
        "soca.app.engine.get_voice_runtime_profile",
        lambda _key: fake_profile,
    )
    monkeypatch.setattr(
        "soca.app.profiles.asr_readiness",
        lambda _selection: SimpleNamespace(ok=True, status="ok", detail="ready"),
    )

    capture = ProtocolCapture()

    def fail_save(_profile: str) -> None:
        raise PermissionError("read-only config")

    instance = SocaEngine(
        voice_config=make_voice_config(),
        text_config=make_text_config(),
        profile="baseline",
        writer=_ProtocolWriter(capture),
        warmup_voice=False,
        voice_profile_saver=fail_save,
    )

    instance._cmd_voice_profile_select({"profile": "qwen-release"})

    assert instance.profile == "baseline"
    assert instance.voice_config is not None
    assert instance.voice_config.profile_key == "baseline"
    assert any(
        event.get("code") == "voice_profile_persist_failed"
        for event in capture.events()
    )


def _stream_of(result: RuntimeResult):
    """Mirror AssistantRuntime._emit_fixed_result for a non-streaming double."""
    if result.response_text:
        yield RuntimeStreamEvent(type="sentence", text=result.response_text)
    yield RuntimeStreamEvent(type="result", result=result)


class _FakeAssistantRuntime:
    def run_text_turn(self, text: str, *, source: str, metadata: dict) -> RuntimeResult:
        return RuntimeResult(response_text=f"echo: {text}", route=RuntimeRoute.FREE_CHAT)

    def stream_text_turn(self, text: str, *, source: str, metadata: dict, **kwargs):
        del kwargs
        yield from _stream_of(self.run_text_turn(text, source=source, metadata=metadata))


class _StreamingAssistantRuntime:
    """Emits several guardrail-passed chunks, like a real free-chat turn."""

    CHUNKS = ("Protein giữ cơ bắp [K1]. ", "Nó tạo cảm giác no [K2]. ", "Vậy nhé.")
    CLEANED = ("Protein giữ cơ bắp. ", "Nó tạo cảm giác no. ", "Vậy nhé.")

    def run_text_turn(self, text: str, *, source: str, metadata: dict) -> RuntimeResult:
        del text, source, metadata
        return RuntimeResult(response_text="".join(self.CHUNKS), route=RuntimeRoute.FREE_CHAT)

    def stream_text_turn(self, text: str, *, source: str, metadata: dict, **kwargs):
        del kwargs
        for chunk in self.CHUNKS:
            yield RuntimeStreamEvent(type="sentence", text=chunk)
        yield RuntimeStreamEvent(
            type="result",
            result=self.run_text_turn(text, source=source, metadata=metadata),
        )


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

    def stream_text_turn(self, text: str, *, source: str, metadata: dict, **kwargs):
        del kwargs
        yield from _stream_of(self.run_text_turn(text, source=source, metadata=metadata))


class _CitedAssistantRuntime:
    def run_text_turn(self, text: str, *, source: str, metadata: dict) -> RuntimeResult:
        del text, source, metadata
        return RuntimeResult(
            response_text="Attention dùng query, key và value [K1].",
            route=RuntimeRoute.KNOWLEDGE_LLM,
            citations=(
                KnowledgeCitation(
                    path="wiki/learning/attention.md",
                    title="Attention",
                    line_start=12,
                    line_end=18,
                ),
            ),
        )

    def stream_text_turn(self, text: str, *, source: str, metadata: dict, **kwargs):
        del kwargs
        yield from _stream_of(self.run_text_turn(text, source=source, metadata=metadata))


class _FailingAssistantRuntime:
    def run_text_turn(self, text: str, *, source: str, metadata: dict) -> RuntimeResult:
        raise RuntimeError("synthetic runtime failure")


def _fake_text_builder(
    config: TextRuntimeConfig, *, session_memory=None, **_runtime_dependencies
) -> TextRuntimeBundle:
    return TextRuntimeBundle(
        runtime=_FakeAssistantRuntime(),  # type: ignore[arg-type]
        session_memory=session_memory,
        llm_status="fake",
        knowledge_status="fake",
        memory_status="fake",
    )


def _manifest_text_builder(
    config: TextRuntimeConfig, *, session_memory=None, **_runtime_dependencies
) -> TextRuntimeBundle:
    return TextRuntimeBundle(
        runtime=_ManifestAssistantRuntime(),  # type: ignore[arg-type]
        session_memory=session_memory,
        llm_status="fake",
        knowledge_status="fake",
        memory_status="fake",
    )


def _cited_text_builder(
    config: TextRuntimeConfig, *, session_memory=None, **_runtime_dependencies
) -> TextRuntimeBundle:
    return TextRuntimeBundle(
        runtime=_CitedAssistantRuntime(),  # type: ignore[arg-type]
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


def test_engine_emits_clean_answer_and_structured_sources() -> None:
    capture = ProtocolCapture()
    code = run_engine(
        voice_config=None,
        text_config=make_text_config(),
        profile="baseline",
        stdin=_commands(capture, {"cmd": "chat", "text": "attention"}, '"done"'),
        stdout=capture,
        text_runtime_builder=_cited_text_builder,
    )

    assert code == 0
    done = next(
        event
        for event in capture.events()
        if event["event"] == "chat" and event["type"] == "done"
    )
    assert done["text"] == "Attention dùng query, key và value."
    assert done["citations"] == [
        {
            "label": "K1",
            "path": "wiki/learning/attention.md",
            "title": "Attention",
            "line_start": 12,
            "line_end": 18,
            "source": "knowledge",
        }
    ]


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

    def failing_builder(
        config: TextRuntimeConfig, *, session_memory=None, **_runtime_dependencies
    ) -> TextRuntimeBundle:
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
    workflow = [
        event
        for event in capture.events()
        if event["event"] in {"turn_started", "turn_terminal"}
    ]
    assert workflow[-1]["event"] == "turn_terminal"
    assert workflow[-1]["payload"]["terminal_status"] == "system_failure"


def test_text_runtime_builder_programming_type_error_is_not_retried() -> None:
    from soca.app.engine import SocaEngine, _ProtocolWriter

    calls = 0

    def broken_builder(config, **dependencies):
        nonlocal calls
        del config, dependencies
        calls += 1
        raise TypeError("internal builder bug")

    instance = SocaEngine(
        voice_config=None,
        text_config=make_text_config(),
        profile="baseline",
        writer=_ProtocolWriter(ProtocolCapture()),
        text_runtime_builder=broken_builder,
        warmup_voice=False,
    )

    with pytest.raises(TypeError, match="internal builder bug"):
        instance._ensure_text_bundle()

    assert calls == 1


@dataclass
class _FakeASR:
    confidence_guard_status: str = "disabled:test"

    def close(self) -> None:
        return None


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


def _fake_voice_builder(
    config: ResolvedVoiceRuntimeConfig, *, session_memory=None
) -> VoiceRuntimeBundle:
    del session_memory
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
        captured["voice"] = kwargs
        return _fake_voice_builder(config)

    def fake_build_text_runtime(config, **kwargs):
        captured["text"] = kwargs
        return _fake_text_builder(config, **kwargs)

    monkeypatch.setattr("soca.app.engine.build_voice_runtime", fake_build_voice_runtime)

    from soca.app.engine import SocaEngine, _ProtocolWriter

    instance = SocaEngine(
        voice_config=make_voice_config(),
        text_config=make_text_config(),
        profile="baseline",
        writer=_ProtocolWriter(ProtocolCapture()),
        text_runtime_builder=fake_build_text_runtime,
        voice_player=_FakeAudioSink(),
        warmup_voice=False,
        llm_engine_factory=lambda *args, **kwargs: object(),
    )
    instance._ensure_text_bundle()
    controller = instance._ensure_voice_controller()
    controller._build_runtime_bundle()

    text_dependencies = captured["text"]
    voice_dependencies = captured["voice"]
    assert isinstance(text_dependencies, dict)
    assert isinstance(voice_dependencies, dict)
    for dependencies in (text_dependencies, voice_dependencies):
        assert dependencies["llm_settings"] is instance.llm_settings
        assert dependencies["secret_store"] is instance.secret_store
        assert dependencies["active_goal_store"] is instance.active_goal_store
        assert dependencies["engine_factory"] is instance.llm_engine_factory


def test_knowledge_index_failure_cleans_up_before_engine_shutdown(
    monkeypatch, tmp_path: Path
) -> None:
    capture = ProtocolCapture()
    monkeypatch.setattr(
        "soca.app.engine.load_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            FileNotFoundError("embedding model missing")
        ),
    )

    def commands():
        yield json.dumps({"cmd": "knowledge_init"}) + "\n"
        capture.wait_for('"action": "init"')
        yield json.dumps({"cmd": "knowledge_index"}) + "\n"
        capture.wait_for('"status": "failed"')
        yield '{"cmd": "quit"}\n'

    code = run_engine(
        voice_config=None,
        text_config=dataclasses.replace(make_text_config(), vault=tmp_path),
        profile="baseline",
        stdin=commands(),
        stdout=capture,
        warmup_voice=False,
    )

    assert code == 0
    events = capture.events()
    assert any(event["event"] == "bye" for event in events)
    failure = next(
        event
        for event in events
        if event.get("event") == "knowledge_setup" and event.get("status") == "failed"
    )
    assert failure["error_code"] == "embedding_model_missing"


def test_knowledge_index_emits_structured_progress_events(monkeypatch, tmp_path: Path) -> None:
    from types import SimpleNamespace

    from soca.knowledge.indexing.coordinator import IndexBuildProgress

    class FakeCoordinator:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def build_blocking(self, *, on_progress, **kwargs):
            del kwargs
            on_progress(IndexBuildProgress("embedding", 3, 5, 2, 1, 1, 5))
            return SimpleNamespace(
                sparse=SimpleNamespace(revision=9),
                dense=SimpleNamespace(reused_rows=2, embedded_rows=3),
            )

        def status(self):
            return SimpleNamespace(
                as_dict=lambda: {
                    "documents": 1,
                    "chunks": 5,
                    "dense_state": "ready",
                }
            )

    monkeypatch.setattr("soca.app.engine.load_model", lambda *args, **kwargs: object())
    monkeypatch.setattr("soca.app.engine.IndexCoordinator", FakeCoordinator)

    capture = ProtocolCapture()

    def commands():
        yield json.dumps({"cmd": "knowledge_init"}) + "\n"
        capture.wait_for('"action": "init"')
        yield json.dumps({"cmd": "knowledge_index"}) + "\n"
        capture.wait_for('"phase": "embedding"')
        capture.wait_for('"status": "ready"')
        yield '{"cmd": "quit"}\n'

    code = run_engine(
        voice_config=None,
        text_config=dataclasses.replace(make_text_config(), vault=tmp_path),
        profile="baseline",
        stdin=commands(),
        stdout=capture,
        warmup_voice=False,
    )

    assert code == 0
    events = [event for event in capture.events() if event.get("event") == "knowledge_setup"]
    progress = next(event for event in events if event.get("phase") == "embedding")
    assert progress["status"] == "running"
    assert progress["completed_chunks"] == 3
    assert progress["total_chunks"] == 5
    assert progress["reused_chunks"] == 2
    assert progress["embedded_chunks"] == 1
    completed = next(
        event
        for event in events
        if event.get("action") == "index" and event.get("status") == "ready"
    )
    assert completed["phase"] == "complete"
    assert completed["documents"] == 1
    assert completed["chunks"] == 5


def test_knowledge_index_start_failure_releases_lock_and_reports_error(
    monkeypatch, tmp_path: Path
) -> None:
    class Writer:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def emit(self, event: dict) -> None:
            self.events.append(event)

    class Config:
        vault = tmp_path

    def fail_start(self) -> None:
        raise RuntimeError("thread start failed")

    engine = object.__new__(SocaEngine)
    engine.text_config = Config()
    engine.writer = Writer()
    engine._knowledge_job_lock = threading.Lock()
    engine._knowledge_job_thread = None
    (tmp_path / "wiki").mkdir()
    monkeypatch.setattr(threading.Thread, "start", fail_start)

    engine._cmd_knowledge_index()

    assert engine._knowledge_job_thread is None
    assert engine._knowledge_job_lock.acquire(blocking=False)
    failure = next(
        event for event in engine.writer.events if event.get("event") == "knowledge_setup"
        and event.get("status") == "failed"
    )
    assert failure["status"] == "failed"
    assert failure["error_code"] == "knowledge_index_start_failed"


def test_shutdown_emits_bye_after_worker_cleanup_timeout() -> None:
    class Writer:
        def __init__(self) -> None:
            self.events: list[dict] = []

        def emit(self, event: dict) -> None:
            self.events.append(event)

    class StuckThread:
        def join(self, timeout: float) -> None:
            return None

        def is_alive(self) -> bool:
            return True

    engine = object.__new__(SocaEngine)
    engine._shutdown = False
    engine.writer = Writer()
    engine._cmd_voice_stop = lambda: False
    engine._dispose_text_bundle = lambda: True
    engine._chat_thread = None
    engine._voice_threads = []
    engine._knowledge_job_thread = StuckThread()
    engine._catalog_lock = threading.Lock()
    engine._catalog_threads = set()
    engine.text_bundle = None
    engine.session_memory = None

    engine.shutdown()

    assert engine._shutdown is True
    assert engine.writer.events[-1] == {"event": "bye"}
    assert any(
        event.get("code") == "knowledge_index_stop_timeout"
        for event in engine.writer.events
    )


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


def _streaming_text_builder(
    config: TextRuntimeConfig, *, session_memory=None, **_runtime_dependencies
) -> TextRuntimeBundle:
    return TextRuntimeBundle(
        runtime=_StreamingAssistantRuntime(),  # type: ignore[arg-type]
        session_memory=session_memory,
        llm_status="fake",
        knowledge_status="fake",
        memory_status="fake",
    )


def test_engine_chat_emits_one_answer_delta_per_streamed_chunk() -> None:
    """A UI that animates composing needs the answer to arrive in pieces.

    The chat turn used to call the blocking run_text_turn and emit a single
    answer_delta holding the whole response, so any per-chunk animation
    finished before it started.
    """
    capture = ProtocolCapture()
    code = run_engine(
        voice_config=None,
        text_config=make_text_config(),
        profile="baseline",
        stdin=_commands(capture, {"cmd": "chat", "text": "xin chào"}, '"done"'),
        stdout=capture,
        text_runtime_builder=_streaming_text_builder,
    )

    assert code == 0
    deltas = [event for event in capture.events() if event["event"] == "answer_delta"]
    assert [delta["payload"]["text"] for delta in deltas] == list(
        _StreamingAssistantRuntime.CLEANED
    ), "a delta must not publish citation labels the final answer strips"
    ordered = [
        event["event"]
        for event in capture.events()
        if event["event"] in {"answer_delta", "turn_terminal"}
    ]
    assert ordered[-1] == "turn_terminal", "every delta must precede the terminal"
    done = [e for e in capture.events() if e["event"] == "chat" and e["type"] == "done"]
    assert done[0]["text"] == "".join(_StreamingAssistantRuntime.CLEANED).strip(), (
        "concatenating the deltas must reproduce the authoritative answer"
    )


def _memory_token_voice_builder(
    config: ResolvedVoiceRuntimeConfig, *, session_memory=None
) -> VoiceRuntimeBundle:
    """A voice turn whose streamed tokens carry a memory citation label."""
    del session_memory
    pipeline = _FakePipeline(
        [
            StreamingEvent(type="asr", text="tôi đã ghi gì"),
            StreamingEvent(type="llm_token", text="Bạn đã ghi "),
            StreamingEvent(type="llm_token", text="lịch họp [M1]."),
            StreamingEvent(
                type="done",
                text="Bạn đã ghi lịch họp.",
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


def test_voice_answer_delta_strips_citation_labels_like_the_chat_turn() -> None:
    """Both surfaces publish the same text; only the chat one used to be cleaned.

    The voice deltas come from raw `llm_token` events. A smalltalk turn still
    receives memory context, and MEMORY_GROUNDING_INSTRUCTIONS teaches the model
    to write [M1], so a label can reach a caption that `done` then strips.
    """
    capture = ProtocolCapture()
    code = run_engine(
        voice_config=make_voice_config(),
        text_config=make_text_config(),
        profile="baseline",
        stdin=_commands(capture, {"cmd": "voice_start", "max_turns": 1}, "loop_stopped"),
        stdout=capture,
        voice_runtime_builder=_memory_token_voice_builder,
        voice_recorder=_fake_recorder,
        voice_player=_FakeAudioSink(),
        warmup_voice=False,
    )

    assert code == 0
    deltas = [event for event in capture.events() if event["event"] == "answer_delta"]
    assert [delta["payload"]["text"] for delta in deltas] == [
        "Bạn đã ghi ",
        "lịch họp.",
    ]
