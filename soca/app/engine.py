from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from queue import Queue
from typing import Any, Protocol, TextIO
from uuid import uuid4

import numpy as np

from soca.app.text_runtime import (
    TextRuntimeBundle,
    TextRuntimeConfig,
    build_text_runtime,
    normalize_text_turn,
)
from soca.app.voice_controller import (
    VoiceMonitorController,
    VoiceMonitorEvent,
    VoiceRecorder,
    VoiceRuntimeBuilder,
)
from soca.config import DEFAULT_SETTINGS, LlmSettings, SecretStore, load_settings, save_settings
from soca.core import AudioSink, ResolvedVoiceRuntimeConfig
from soca.core.answer_validation import (
    answer_text_without_citation_labels,
    citation_records,
)
from soca.core.context_budget import (
    PromptAssembler,
    PromptBudgetError,
    PromptComponent,
    Utf8TokenCounter,
    capability_from_values,
)
from soca.core.profiles import get_voice_runtime_profile
from soca.core.usage import SessionUsage, TurnUsage
from soca.core.voice_runtime import build_voice_runtime
from soca.core.workflow import (
    ActiveGoalStore,
    EventStatus,
    EventType,
    GoalCheckpointStore,
    GoalStatus,
    TerminalOutcome,
    TerminalStatus,
    TurnNode,
    WorkflowEventStream,
)
from soca.core.workflow.protocol import (
    CURRENT_PROTOCOL_VERSION,
    protocol_hello,
    workflow_event_to_protocol,
)
from soca.core.workflow.runtime_events import terminal_from_runtime_result
from soca.knowledge.index.persistence import default_index_home
from soca.knowledge.indexing.coordinator import IndexCoordinator
from soca.knowledge.indexing.identity import CorpusSpec
from soca.knowledge.indexing.models import load_model
from soca.knowledge.markdown_vault import MarkdownVaultKnowledgeSource
from soca.knowledge.vault import init_knowledge_vault
from soca.llm.factory import DEFAULT_LLM_ENGINE_FACTORY, EngineBuilder
from soca.llm.providers import (
    PRICING_TABLE_AS_OF,
    LLMProvider,
    RemoteLLMError,
    RemoteModelInfo,
    fetch_catalog,
    get_provider,
    search_models,
)
from soca.llm.registry import LLM_MODEL_REGISTRY
from soca.memory import (
    CoreMemoryStore,
    MemoryCommands,
    ProposalStore,
    SessionCheckpointStore,
    SessionMemory,
    default_session_checkpoint_home,
)
from soca.prompts import SOCA_RUNTIME_SYSTEM_PROMPT
from soca.tts import VALTEC_TTS_CONFIG

PROTOCOL_VERSION = CURRENT_PROTOCOL_VERSION
LOGGER = logging.getLogger(__name__)



class TextRuntimeBuilder(Protocol):
    def __call__(
        self,
        config: TextRuntimeConfig,
        *,
        session_memory: SessionMemory | None,
        llm_settings: LlmSettings,
        secret_store: LlmSecretStore,
        engine_factory: EngineBuilder,
        active_goal_store: ActiveGoalStore,
    ) -> TextRuntimeBundle: ...


SettingsLoader = Callable[[], LlmSettings]
SettingsSaver = Callable[[LlmSettings], None]
CatalogFetcher = Callable[[LLMProvider, str], list[RemoteModelInfo]]


@dataclasses.dataclass
class _TurnProgressContext:
    surface: str
    run_id: str
    goal_id: str
    workflow: WorkflowEventStream
    sequence: int = 0


def _memory_protocol_mode(
    mode: object,
    degraded_reason: object,
    hit_count: object,
) -> str:
    if degraded_reason or mode == "degraded":
        return "degraded"
    if mode == "retrieved":
        return "retrieved"
    if mode == "degraded":
        return "degraded"
    return "none"


def _progress_phase_for_stage(stage: str) -> str:
    """Collapse internal runtime stages into stable UI-facing phases."""

    if stage == "input_guardrail":
        return "analyzing"
    if stage == "tool_router":
        return "routing"
    if stage in {"memory_context", "memory_archive_context"} or stage.startswith("tool:memory."):
        return "memory"
    if stage == "knowledge_context" or stage.startswith("tool:knowledge."):
        return "retrieval"
    if stage.startswith("tool:"):
        return "tool"
    if stage == "llm":
        return "synthesis"
    if stage.startswith("evidence_completion:"):
        return "validation"
    if stage in {
        "output_guardrail",
        "tool_input_guardrail",
        "tool_output_guardrail",
    }:
        return "validation"
    return "analyzing"


def _terminal_status_for_result(result: Any) -> str:
    route = getattr(getattr(result, "route", None), "value", "")
    if route == "clarification":
        return "needs_clarification"
    trace = getattr(result, "trace", None)
    if getattr(trace, "evidence_completion_status", "") == "budget_exhausted":
        return "budget_exhausted"
    if getattr(trace, "evidence_completion_status", "") == "insufficient":
        return "insufficient_evidence"
    if getattr(trace, "evidence_status", "") == "insufficient":
        return "insufficient_evidence"
    if bool(getattr(result, "blocked", False)):
        return "safe_failure"
    return "achieved"


def _workflow_node_for_phase(phase: str) -> TurnNode:
    return {
        "preparing": TurnNode.ADMIT,
        "analyzing": TurnNode.RESOLVE_GOAL,
        "routing": TurnNode.CHOOSE_CAPABILITY,
        "memory": TurnNode.EXECUTE_ACTION,
        "retrieval": TurnNode.EXECUTE_ACTION,
        "tool": TurnNode.EXECUTE_ACTION,
        "synthesis": TurnNode.SYNTHESIZE,
        "validation": TurnNode.VERIFY_ANSWER,
        "speech": TurnNode.FINALIZE,
        "complete": TurnNode.FINALIZE,
    }.get(phase, TurnNode.ADMIT)


def _retrieval_trace_payload(
    result: Any, trace: Any, hits: tuple[Any, ...] | list[Any]
) -> dict[str, Any]:
    columns: dict[str, list[dict[str, Any]]] = {}
    fused: list[dict[str, Any]] = []
    for hit in hits:
        backend = str(getattr(hit, "retrieval_backend", "unknown") or "unknown")
        item: dict[str, Any] = {
            "path": str(getattr(getattr(hit, "document", None), "path", ""))[:240],
            "score": float(getattr(hit, "score", 0.0)),
        }
        for field in ("sparse_score", "dense_score", "fusion_score"):
            value = getattr(hit, field, None)
            if value is not None:
                item[field] = float(value)
        columns.setdefault(backend, []).append(item)
        fused.append(
            {
                "path": item["path"],
                "picked": True,
                "backend": backend,
                "score": item["score"],
            }
        )
    decision = next(
        (
            item
            for item in getattr(trace, "evidence_decisions", ())
            if getattr(item, "source", None) == "knowledge"
        ),
        None,
    )
    return {
        "event": "retrieval_trace",
        "query": getattr(getattr(result, "frame", None), "text", ""),
        "tier": (
            trace.tool_router_tier
            if trace.tool_router_tier in {"deterministic", "semantic", "llm", "none"}
            else "none"
        ),
        "latency_ms": trace.stage_latencies_ms.get("knowledge", 0.0),
        "columns": [
            {"source": source, "hits": source_hits}
            for source, source_hits in columns.items()
        ],
        "fused": fused,
        "rejected_count": int(getattr(decision, "rejected_count", 0) or 0),
        "evidence": decision.as_dict() if decision is not None else None,
    }


class LlmSecretStore(Protocol):
    def get_key(self, provider_key: str) -> str | None: ...

    def set_key(self, provider_key: str, value: str) -> None: ...

    def has_key(self, provider_key: str) -> bool: ...

    @staticmethod
    def mask(value: str) -> str: ...


def _sanitize(value: Any) -> Any:
    """Coerce an event payload into JSON-safe data, never raising."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {"ndarray": list(value.shape)}
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _sanitize(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(v) for v in value]
    return str(value)


class _ProtocolWriter:
    """Serialized NDJSON writer; safe to call from worker threads."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._lock = threading.Lock()

    def emit(self, payload: dict[str, Any]) -> None:
        line = json.dumps(_sanitize(payload), ensure_ascii=False)
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()


def _model_protocol_payload(model: RemoteModelInfo) -> dict[str, Any]:
    """Expose only the catalog fields consumed by the external UI.

    ``RemoteModelInfo.supported_parameters`` is useful to the LLM adapter, but
    it is not part of the UI protocol. Omitting it keeps the OpenRouter catalog
    event compact enough for a single NDJSON frame.
    """
    return {
        "id": model.id,
        "label": model.label,
        "context_length": model.context_length,
        "price_prompt_per_1m": model.price_prompt_per_1m,
        "price_completion_per_1m": model.price_completion_per_1m,
        "pricing_source": model.pricing_source,
        "max_output_tokens": model.max_output_tokens,
        "reasoning_supported": model.reasoning_supported,
        "reasoning_mandatory": model.reasoning_mandatory,
    }


class SocaEngine:
    """Command dispatcher bridging the protocol to the existing runtimes."""

    def __init__(
        self,
        *,
        voice_config: ResolvedVoiceRuntimeConfig | None,
        text_config: TextRuntimeConfig,
        profile: str,
        no_model: bool = False,
        writer: _ProtocolWriter,
        text_runtime_builder: TextRuntimeBuilder = build_text_runtime,
        voice_runtime_builder: VoiceRuntimeBuilder | None = None,
        voice_recorder: VoiceRecorder | None = None,
        voice_player: AudioSink | None = None,
        warmup_voice: bool = True,
        llm_settings_loader: SettingsLoader = load_settings,
        llm_settings_saver: SettingsSaver = save_settings,
        secret_store: LlmSecretStore | None = None,
        catalog_fetcher: CatalogFetcher = fetch_catalog,
        llm_engine_factory: EngineBuilder = DEFAULT_LLM_ENGINE_FACTORY,
    ) -> None:
        self.voice_config = voice_config
        self.text_config = text_config
        self.profile = profile
        self.no_model = no_model
        self.writer = writer
        self.text_runtime_builder = text_runtime_builder
        self.voice_runtime_builder = voice_runtime_builder
        self.voice_recorder = voice_recorder
        self.voice_player = voice_player
        self.warmup_voice = warmup_voice
        self.llm_settings_saver = llm_settings_saver
        self._settings_error: str | None = None
        try:
            self.llm_settings = llm_settings_loader()
        except ValueError:
            LOGGER.warning("Persisted LLM settings are invalid; runtime is not ready", exc_info=True)
            self.llm_settings = DEFAULT_SETTINGS
            self._settings_error = (
                "Cấu hình LLM đã lưu không hợp lệ; runtime bị khóa cho đến khi bạn lưu cấu hình mới."
            )
        self.secret_store = secret_store or SecretStore()
        self.catalog_fetcher = catalog_fetcher
        self.llm_engine_factory = llm_engine_factory
        # Provider /models endpoints (especially OpenRouter) can be large and
        # slow. Never run them on the stdin command loop: doing so makes a key
        # submit, startup, or model picker look frozen.
        self._catalog_lock = threading.Lock()
        self._catalog_cache: dict[str, tuple[str, list[RemoteModelInfo]]] = {}
        self._catalog_inflight: set[tuple[str, str]] = set()
        self._catalog_threads: set[threading.Thread] = set()
        self._key_validation_tokens: dict[str, str] = {}
        self._knowledge_job_lock = threading.Lock()
        self._knowledge_job_thread: threading.Thread | None = None

        self.session_memory = self._create_session_memory()
        self.active_goal_store = ActiveGoalStore(
            checkpoint_store=(
                GoalCheckpointStore(default_session_checkpoint_home() / "goals")
                if self.text_config.session_persistence == "local_resumable"
                else None
            ),
            session_id=self.text_config.session_id,
        )
        self.text_bundle: TextRuntimeBundle | None = None
        self.voice_controller: VoiceMonitorController | None = None
        self.voice_stop_event: threading.Event | None = None
        self._voice_threads: list[threading.Thread] = []
        self._chat_thread: threading.Thread | None = None
        self._chat_lock = threading.Lock()
        # Session usage accumulates across chat AND voice turns (shared session).
        self.session_usage = SessionUsage()
        self._usage_lock = threading.Lock()
        self._last_prompt_manifest: dict[str, Any] | None = None
        self._progress_lock = threading.Lock()
        self._progress_contexts: dict[str, _TurnProgressContext] = {}
        self._shutdown = False

    # --- lifecycle --------------------------------------------------------------

    def hello(self) -> None:
        stack: dict[str, Any] = {"llm": self.text_config.llm_model}
        if self.voice_config is not None:
            selected = self._selected_voice_settings()
            llm_label = (
                f"{selected.provider_key}:{selected.model_id}"
                if selected.backend == "remote"
                else selected.model_id
            )
            stack = {
                "asr": self.voice_config.asr_model,
                "llm": llm_label,
                "tts": VALTEC_TTS_CONFIG.key,
                "voice": self.voice_config.tts_voice,
            }
        self.writer.emit(protocol_hello(profile=self.profile, no_model=self.no_model, stack=stack))
        self._cmd_context()
        if self._settings_error is not None:
            self._error(self._settings_error, code="llm_settings_invalid")

    def dispatch(self, command: dict[str, Any]) -> bool:
        """Handle one command; return False when the engine should exit."""
        cmd = command.get("cmd")
        if cmd == "quit":
            return False
        if cmd == "status":
            self._cmd_status()
        elif cmd == "context":
            self._cmd_context()
        elif cmd == "memory":
            self._cmd_memory()
        elif cmd == "memory_compact":
            self._cmd_memory_compact(str(command.get("action") or "request"))
        elif cmd == "memory_proposals":
            self._cmd_memory_proposals()
        elif cmd == "memory_approve":
            self._cmd_memory_action(command, "approve")
        elif cmd == "memory_reject":
            self._cmd_memory_action(command, "reject")
        elif cmd == "usage":
            self._cmd_usage()
        elif cmd == "llm_providers":
            self._cmd_llm_providers()
        elif cmd == "llm_models":
            self._cmd_llm_models(command)
        elif cmd == "llm_set_key":
            self._cmd_llm_set_key(command)
        elif cmd == "llm_select":
            self._cmd_llm_select(command)
        elif cmd == "llm_config":
            self._emit_llm_config()
        elif cmd == "chat":
            self._cmd_chat(str(command.get("text") or ""))
        elif cmd == "voice_start":
            max_turns = command.get("max_turns")
            self._cmd_voice_start(int(max_turns) if isinstance(max_turns, int) else None)
        elif cmd == "voice_stop":
            self._cmd_voice_stop()
        elif cmd == "voice_profile_select":
            self._cmd_voice_profile_select(command)
        elif cmd == "knowledge_init":
            self._cmd_knowledge_init()
        elif cmd == "knowledge_index":
            self._cmd_knowledge_index()
        else:
            self._error(f"unknown command: {cmd!r}")
        return True

    def shutdown(self) -> None:
        if self._shutdown:
            return
        cleanup_failed = self._cmd_voice_stop()
        if not self._dispose_text_bundle():
            cleanup_failed = True
        if self._chat_thread is not None:
            self._chat_thread.join(timeout=10.0)
            if self._chat_thread.is_alive():
                cleanup_failed = True
                self._error("chat thread did not stop", code="chat_thread_stop_timeout")
        for thread in self._voice_threads:
            thread.join(timeout=5.0)
            if thread.is_alive():
                cleanup_failed = True
                self._error("voice thread did not stop", code="voice_thread_stop_timeout")
        knowledge_thread = self._knowledge_job_thread
        if knowledge_thread is not None:
            knowledge_thread.join(timeout=30.0)
            if knowledge_thread.is_alive():
                cleanup_failed = True
                self._error(
                    "knowledge index thread did not stop",
                    code="knowledge_index_stop_timeout",
                )
        # Fake fetchers finish immediately in tests; real HTTP fetches are
        # bounded below the UI termination grace period. Threads are daemonized
        # so a slow provider cannot keep the process alive forever.
        with self._catalog_lock:
            catalog_threads = tuple(self._catalog_threads)
        for thread in catalog_threads:
            thread.join(timeout=1.0)
            if thread.is_alive():
                cleanup_failed = True
                self._error("model catalog fetch did not stop", code="catalog_thread_stop_timeout")
        if self.text_bundle is not None:
            if not self._dispose_text_bundle():
                cleanup_failed = True
        if self.session_memory is not None:
            try:
                self.session_memory.close()
            except Exception as exc:  # noqa: BLE001 - expose cleanup failure
                cleanup_failed = True
                self._error(
                    "session memory cleanup failed",
                    code="session_cleanup_failed",
                    detail=type(exc).__name__,
                )
        if cleanup_failed:
            return
        self._shutdown = True
        self.writer.emit({"event": "bye"})

    def _dispose_text_bundle(self) -> bool:
        """Close the current text runtime before a configuration mutation.

        A runtime owns the provider engine and retrieval watchers it built.  A
        configuration refresh must not discard that owner without closing it;
        otherwise every model/key change leaks a provider client and index
        watcher.  Keep the bundle attached when close fails so a later
        operator retry can observe and repeat the failed cleanup.
        """
        bundle = self.text_bundle
        if bundle is None:
            return True
        try:
            bundle.close()
        except Exception as exc:  # noqa: BLE001 - lifecycle boundary
            self._error(
                "text runtime cleanup failed",
                code="runtime_cleanup_failed",
                detail=type(exc).__name__,
            )
            return False
        self.text_bundle = None
        return True

    def _error(self, message: str, **extra: Any) -> None:
        self.writer.emit({"event": "engine_error", "message": message, **extra})

    def _emit_knowledge_setup(self, action: str, status: str, detail: str, **extra: Any) -> None:
        self.writer.emit(
            {
                "event": "knowledge_setup",
                "action": action,
                "status": status,
                "vault": str(self.text_config.vault),
                "detail": detail,
                **extra,
            }
        )

    def _cmd_knowledge_init(self) -> None:
        try:
            result = init_knowledge_vault(self.text_config.vault)
        except (OSError, ValueError) as exc:
            self._emit_knowledge_setup(
                "init",
                "failed",
                str(exc),
                error_code="knowledge_init_failed",
            )
            return
        self._emit_knowledge_setup(
            "init",
            "ready",
            f"{len(result.created_dirs)} thư mục mới · {len(result.created_files)} file mới",
            created_dirs=len(result.created_dirs),
            created_files=len(result.created_files),
            skipped_files=len(result.skipped_files),
        )
        self._cmd_status()

    def _cmd_knowledge_index(self) -> None:
        vault = self.text_config.vault
        if not (vault / "wiki").is_dir():
            self._emit_knowledge_setup(
                "index",
                "failed",
                "Vault chưa được init; hãy chạy Init vault trước.",
                error_code="knowledge_not_initialized",
            )
            return
        if not self._knowledge_job_lock.acquire(blocking=False):
            self._emit_knowledge_setup(
                "index",
                "busy",
                "Index đang chạy.",
                error_code="knowledge_index_busy",
            )
            return

        def run() -> None:
            try:
                self._emit_knowledge_setup("index", "running", "Đang quét và tạo embedding…")
                reader = MarkdownVaultKnowledgeSource(
                    vault,
                    include_globs=("wiki/**/*.md",),
                )
                spec = CorpusSpec(vault_path=vault)
                coordinator = IndexCoordinator(
                    reader,
                    spec=spec,
                    index_home=default_index_home(vault),
                    model=load_model("aiteamvn-v2", allow_download=False),
                )
                report = coordinator.build_blocking(dense=True, verify_content=True)
                status = coordinator.status().as_dict()
                self._emit_knowledge_setup(
                    "index",
                    "ready",
                    f"Đã index {status['documents']} tài liệu · {status['chunks']} chunks.",
                    revision=report.sparse.revision,
                    documents=status["documents"],
                    chunks=status["chunks"],
                    dense_state=status["dense_state"],
                )
                self._cmd_status()
            except ImportError as exc:
                self._emit_knowledge_setup(
                    "index",
                    "failed",
                    str(exc),
                    error_code="embedding_dependency_missing",
                )
            except FileNotFoundError as exc:
                self._emit_knowledge_setup(
                    "index",
                    "failed",
                    str(exc),
                    error_code="embedding_model_missing",
                )
            except OSError as exc:
                self._emit_knowledge_setup(
                    "index",
                    "failed",
                    str(exc),
                    error_code="knowledge_index_io_error",
                )
            except RuntimeError as exc:
                self._emit_knowledge_setup(
                    "index",
                    "failed",
                    str(exc),
                    error_code="knowledge_index_runtime_error",
                )
            except ValueError as exc:
                self._emit_knowledge_setup(
                    "index",
                    "failed",
                    str(exc),
                    error_code="knowledge_index_invalid",
                )
            finally:
                self._knowledge_job_lock.release()
                self._knowledge_job_thread = None

        thread = threading.Thread(
            target=run,
            daemon=True,
            name="soca-knowledge-index",
        )
        self._knowledge_job_thread = thread
        thread.start()

    # --- status -----------------------------------------------------------------

    def _cmd_status(self) -> None:
        from soca.app.profiles import collect_runtime_profile_readiness
        from soca.knowledge.index.persistence import default_index_home
        from soca.knowledge.indexing.catalog import IndexCatalog
        from soca.knowledge.indexing.identity import CorpusSpec
        from soca.knowledge.indexing.models import model_fingerprint, model_is_provisioned

        profiles = [
            {
                "key": item.key,
                "status": item.profile_status,
                "asr": item.asr_model,
                "llm": item.llm_model,
                "tts": item.tts_engine,
                "voice": item.tts_voice,
            }
            for item in collect_runtime_profile_readiness()
        ]
        knowledge_index: dict[str, Any] | None = None
        vault = self.text_config.vault
        knowledge_vault = {
            "path": str(vault),
            "initialized": (vault / "wiki").is_dir()
            and (vault / "memory" / "core.json").is_file(),
            "index_home": str(default_index_home(vault)),
        }
        embedding_ready = False
        try:
            embedding_ready = model_is_provisioned("aiteamvn-v2")
            embedding_fingerprint = (
                model_fingerprint("aiteamvn-v2") if embedding_ready else None
            )
            knowledge_index = (
                IndexCatalog(default_index_home(self.text_config.vault))
                .status(
                    CorpusSpec(vault_path=self.text_config.vault),
                    embedding_fingerprint=embedding_fingerprint,
                )
                .as_dict()
            )
        except (OSError, RuntimeError, ValueError) as exc:
            LOGGER.debug("Could not inspect knowledge index status: %s", exc)
        self.writer.emit(
            {
                "event": "status",
                "profiles": profiles,
                "knowledge_vault": knowledge_vault,
                "knowledge_index": knowledge_index,
                "runtime_components": self._runtime_component_statuses(
                    embedding_ready=embedding_ready,
                    knowledge_index=knowledge_index,
                ),
            }
        )

    def _runtime_component_statuses(
        self,
        *,
        embedding_ready: bool,
        knowledge_index: dict[str, Any] | None,
    ) -> list[dict[str, str]]:
        """Describe configured runtime dependencies without eagerly loading them."""

        from soca.app.profiles import asr_readiness
        from soca.asr.qwen_artifacts import QWEN_RELEASE_ARTIFACT
        from soca.asr.qwen_readiness import inspect_qwen_readiness
        from soca.core.smart_turn import _MODEL_FILE as SMART_TURN_MODEL_FILE
        from soca.llm.registry import get_model_config
        from soca.memory.summary import default_summary_model_root, production_summary_model_spec

        components: list[dict[str, str]] = []

        def add(component_id: str, label: str, status: str, detail: str) -> None:
            components.append(
                {"id": component_id, "label": label, "status": status, "detail": detail}
            )

        settings = self.llm_settings
        if self.no_model:
            add("chat_llm", "Chat LLM", "disabled", "engine started with --no-model")
        elif self._settings_error is not None:
            add("chat_llm", "Chat LLM", "invalid", self._settings_error)
        elif settings.backend == "remote":
            key_state = "ready" if self.secret_store.has_key(settings.provider_key) else "missing"
            add(
                "chat_llm",
                "Chat LLM",
                key_state,
                f"remote · {settings.provider_key}:{settings.model_id}",
            )
        else:
            local_config = get_model_config(settings.model_id)
            local_state = (
                "loaded"
                if self.text_bundle is not None
                else ("ready" if local_config.local_path.is_file() else "missing")
            )
            add("chat_llm", "Chat LLM", local_state, f"local · {settings.model_id}")

        voice_bundle = self.voice_controller.bundle if self.voice_controller is not None else None
        if self.voice_config is None or self.no_model:
            add("voice_asr", "Voice ASR", "disabled", "voice runtime not configured")
            add("voice_llm", "Voice LLM", "disabled", "voice runtime not configured")
            add("tts", "TTS", "disabled", "voice runtime not configured")
            add("smart_turn", "SmartTurn", "disabled", "adaptive endpoint unavailable")
            add("vad", "VAD", "disabled", "voice runtime not configured")
            add("asr_guards", "ASR guards", "disabled", "voice runtime not configured")
        elif self._settings_error is not None:
            add("voice_asr", "Voice ASR", "blocked", "LLM settings invalid")
            add("voice_llm", "Voice LLM", "invalid", self._settings_error)
            add("tts", "TTS", "blocked", "LLM settings invalid")
            add("smart_turn", "SmartTurn", "blocked", "LLM settings invalid")
            add("vad", "VAD", "blocked", "LLM settings invalid")
            add("asr_guards", "ASR guards", "blocked", "LLM settings invalid")
        else:
            configured_asr = asr_readiness(self.voice_config.asr)
            add(
                "voice_asr",
                "Voice ASR",
                "loaded" if voice_bundle is not None else configured_asr.status,
                (
                    f"{self.voice_config.asr_model} · {voice_bundle.asr_context_status}"
                    if voice_bundle is not None
                    else configured_asr.detail
                ),
            )
            voice_settings = self._selected_voice_settings()
            if voice_settings.backend == "remote":
                voice_llm_ready = self.secret_store.has_key(voice_settings.provider_key)
                voice_llm_state = "loaded" if voice_bundle is not None else (
                    "ready" if voice_llm_ready else "missing"
                )
                voice_llm_detail = (
                    f"remote · {voice_settings.provider_key}:{voice_settings.model_id}"
                )
            else:
                local_config = get_model_config(voice_settings.model_id)
                voice_llm_ready = local_config.local_path.is_file()
                voice_llm_state = "loaded" if voice_bundle is not None else (
                    "ready" if voice_llm_ready else "missing"
                )
                voice_llm_detail = f"local · {voice_settings.model_id}"
            add(
                "voice_llm",
                "Voice LLM",
                voice_llm_state,
                voice_llm_detail,
            )
            try:
                from soca.tts.valtec.artifacts import resolve_current_valtec_release

                resolve_current_valtec_release()
                tts_state = "loaded" if voice_bundle is not None else "ready"
            except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
                tts_state = "missing"
            add("tts", "TTS", tts_state, f"{VALTEC_TTS_CONFIG.key}/{self.voice_config.tts_voice}")

            smart_turn_path = (
                Path(__file__).resolve().parents[2]
                / "models"
                / "smart-turn-v3-onnx"
                / SMART_TURN_MODEL_FILE
            )
            if not self.voice_config.adaptive_endpoint:
                smart_state = "disabled"
            elif voice_bundle is not None and voice_bundle.turn_detector is not None:
                smart_state = "loaded"
            else:
                smart_state = "ready" if smart_turn_path.is_file() else "missing"
            add("smart_turn", "SmartTurn", smart_state, SMART_TURN_MODEL_FILE)
            add(
                "vad",
                "VAD",
                "loaded" if voice_bundle is not None else "configured",
                "Silero VAD · lazy" if voice_bundle is None else "Silero VAD",
            )
            data_asr = Path(__file__).resolve().parents[2] / "data" / "asr"
            calibration_path = data_asr / "threshold_calibration.json"
            guard_state = "ready" if calibration_path.is_file() else "degraded"
            add(
                "asr_guards",
                "ASR guards",
                guard_state,
                f"confidence {'+' if calibration_path.is_file() else '-'}",
            )

        qwen_readiness = inspect_qwen_readiness(QWEN_RELEASE_ARTIFACT)
        add(
            "qwen_asr_release",
            "Qwen ASR release",
            qwen_readiness.state.value,
            qwen_readiness.detail,
        )

        if self.session_memory is None:
            add("summary", "Working summary", "disabled", "session memory disabled")
            add("memory", "Archive memory", "disabled", "session memory disabled")
        else:
            summary_spec = production_summary_model_spec()
            summary_path = summary_spec.path(default_summary_model_root())
            if self.session_memory.summary_model_key is None:
                add("summary", "Working summary", "disabled", "worker disabled")
            elif self.session_memory.summary_worker_state == "running":
                summary_state = "loaded"
                add(
                    "summary",
                    "Working summary",
                    summary_state,
                    f"local · {summary_spec.key} · lazy · checksum on use",
                )
            else:
                summary_state = "ready" if summary_path.is_file() else "missing"
                add(
                    "summary",
                    "Working summary",
                    summary_state,
                    f"local · {summary_spec.key} · lazy · checksum on use",
                )
            add(
                "memory",
                "Archive memory",
                "configured",
                f"retrieved/{self.text_config.memory_retrieval_mode}"
                f" · session {self.session_memory.persistence}",
            )

        embedding_detail = (
            "aiteamvn-v2 · provisioned"
            if embedding_ready
            else "aiteamvn-v2 · not provisioned"
        )
        embedding_state = "ready" if embedding_ready else "missing"
        if knowledge_index is not None:
            embedding_detail += f" · dense {knowledge_index.get('dense_state', 'unknown')}"
        add("embedding", "Embedding", embedding_state, embedding_detail)

        if self.text_config.semantic_router_enabled:
            router_state = "loaded" if self.text_bundle is not None else "configured"
            add(
                "semantic_router",
                "Semantic router",
                router_state,
                f"threshold {self.text_config.semantic_router_threshold:.2f} · lazy",
            )
        else:
            add("semantic_router", "Semantic router", "disabled", "disabled by config")
        add(
            "tool_router",
            "Tool router",
            "loaded" if self.text_bundle is not None else "configured",
            f"text:{self.text_config.tool_router_mode}",
        )
        return components

    # --- remote LLM configuration ---------------------------------------------

    def _cmd_llm_providers(self) -> None:
        from soca.llm.providers import PROVIDER_REGISTRY

        providers = [
            {
                "key": provider.key,
                "label": provider.label,
                "has_key": self.secret_store.has_key(provider.key),
                "has_pricing_api": provider.has_pricing_api,
            }
            for provider in PROVIDER_REGISTRY.values()
        ]
        self.writer.emit({"event": "llm_providers", "providers": providers})

    def _cmd_llm_models(self, command: dict[str, Any]) -> None:
        provider = self._provider_from_command(command)
        if provider is None:
            return
        query = command.get("query", "")
        if not isinstance(query, str):
            self._error("LLM model query phải là chuỗi.")
            return
        api_key = self.secret_store.get_key(provider.key)
        if not api_key:
            self._error(f"Chưa có API key cho {provider.label}.")
            return
        cached = self._cached_catalog(provider, api_key)
        if cached is not None:
            self._emit_catalog(provider, cached, query)
            return
        self._emit_catalog(provider, [], query)
        self._start_catalog_fetch(provider, api_key, purpose="models", query=query)

    def _cmd_llm_set_key(self, command: dict[str, Any]) -> None:
        if self._voice_is_active():
            self._error("Không thể đổi API key khi voice đang chạy; hãy dừng voice trước.")
            return
        provider = self._provider_from_command(command)
        if provider is None:
            return
        value = command.get("key")
        if not isinstance(value, str) or not value.strip():
            self._emit_key_status(provider.key, ok=False, message="API key không được để trống.")
            return
        secret = value.strip()
        fingerprint = self._catalog_fingerprint(secret)
        with self._catalog_lock:
            self._key_validation_tokens[provider.key] = fingerprint
        self._emit_key_status(
            provider.key,
            ok=False,
            pending=True,
            message=f"Đang xác thực API key của {provider.label}…",
        )
        self._start_catalog_fetch(provider, secret, purpose="key")

    def _cmd_llm_select(self, command: dict[str, Any]) -> None:
        if self._chat_lock.locked():
            self._error("Không thể đổi LLM khi lượt chat hiện tại đang chạy.")
            return
        if self._voice_is_active():
            self._error("Không thể đổi LLM khi voice đang chạy; hãy dừng voice trước.")
            return
        backend = command.get("backend")
        if backend not in ("local", "remote"):
            self._error("LLM backend phải là 'local' hoặc 'remote'.")
            return
        provider_key = command.get("provider", self.llm_settings.provider_key)
        model_id = command.get("model", self.llm_settings.model_id)
        if not isinstance(provider_key, str) or not isinstance(model_id, str):
            self._error("LLM provider và model phải là chuỗi.")
            return
        max_tokens = command.get("max_tokens", self.llm_settings.max_tokens)
        reasoning_enabled = command.get("reasoning_enabled", self.llm_settings.reasoning_enabled)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            self._error("Max output tokens phải là số nguyên.")
            return
        if not isinstance(reasoning_enabled, bool):
            self._error("Reasoning phải là true hoặc false.")
            return
        model_info = (
            self._remote_model_info(provider_key, model_id) if backend == "remote" else None
        )
        try:
            settings = LlmSettings(
                backend=backend,
                provider_key=provider_key,
                model_id=model_id,
                max_tokens=max_tokens,
                reasoning_enabled=reasoning_enabled,
                temperature=self.llm_settings.temperature,
                top_p=self.llm_settings.top_p,
                model_context_window=(
                    model_info.context_length if model_info is not None else None
                ),
                model_max_output_tokens=(
                    model_info.max_output_tokens if model_info is not None else None
                ),
                model_reasoning_supported=(
                    model_info.reasoning_supported if model_info is not None else None
                ),
                model_reasoning_mandatory=(
                    model_info.reasoning_mandatory if model_info is not None else False
                ),
                model_reasoning_parameter=(
                    model_info.reasoning_parameter if model_info is not None else None
                ),
            )
        except ValueError as exc:
            self._error(str(exc))
            return
        if settings.backend == "remote" and not self.secret_store.has_key(settings.provider_key):
            provider = get_provider(settings.provider_key)
            self._error(f"Chưa có API key cho {provider.label}.")
            return
        if not self._dispose_text_bundle():
            return
        try:
            self.llm_settings_saver(settings)
        except ValueError as exc:
            self._error(str(exc))
            return
        self.llm_settings = settings
        self._settings_error = None
        self._invalidate_voice_runtime()
        self._last_prompt_manifest = None
        self._emit_llm_config()

    def _emit_llm_config(self) -> None:
        settings = self.llm_settings
        active_model = self._active_remote_model(settings)
        if settings.backend == "remote":
            provider = get_provider(settings.provider_key)
            api_key = self.secret_store.get_key(provider.key)
            if api_key and active_model is None:
                self._start_catalog_fetch(provider, api_key, purpose="config")
        payload: dict[str, Any] = {
            "event": "llm_config",
            "backend": settings.backend,
            "provider": settings.provider_key,
            "model": settings.model_id,
            "max_tokens": settings.max_tokens,
            "effective_max_tokens": settings.effective_max_tokens,
            "reasoning_enabled": settings.reasoning_enabled,
            "effective_reasoning_enabled": settings.effective_reasoning_enabled,
            "reasoning_supported": settings.model_reasoning_supported,
            "reasoning_mandatory": settings.model_reasoning_mandatory,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "pricing_as_of": PRICING_TABLE_AS_OF,
            "pricing": active_model,
            "context_length": self._model_context_length(),
            "runtime_ready": self._settings_error is None,
            "settings_error": self._settings_error,
        }
        self.writer.emit(payload)
        self._cmd_context()

    def _active_remote_model(self, settings: LlmSettings) -> RemoteModelInfo | None:
        if settings.backend != "remote":
            return None
        api_key = self.secret_store.get_key(settings.provider_key)
        if not api_key:
            return None
        provider = get_provider(settings.provider_key)
        catalog = self._cached_catalog(provider, api_key)
        if catalog is None:
            return None
        return next((model for model in catalog if model.id == settings.model_id), None)

    def _remote_model_info(
        self,
        provider_key: str,
        model_id: str,
    ) -> RemoteModelInfo | None:
        api_key = self.secret_store.get_key(provider_key)
        if not api_key:
            return None
        provider = get_provider(provider_key)
        catalog = self._cached_catalog(provider, api_key)
        if catalog is None:
            return None
        return next((model for model in catalog if model.id == model_id), None)

    @staticmethod
    def _catalog_fingerprint(api_key: str) -> str:
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()

    def _cached_catalog(
        self,
        provider: LLMProvider,
        api_key: str,
    ) -> list[RemoteModelInfo] | None:
        fingerprint = self._catalog_fingerprint(api_key)
        with self._catalog_lock:
            cached = self._catalog_cache.get(provider.key)
            if cached is None or cached[0] != fingerprint:
                return None
            return list(cached[1])

    def _emit_catalog(
        self,
        provider: LLMProvider,
        catalog: list[RemoteModelInfo],
        query: str,
    ) -> None:
        self.writer.emit(
            {
                "event": "llm_catalog",
                "provider": provider.key,
                "models": [
                    _model_protocol_payload(model) for model in search_models(catalog, query)
                ],
                "pricing_as_of": PRICING_TABLE_AS_OF,
            }
        )

    def _start_catalog_fetch(
        self,
        provider: LLMProvider,
        api_key: str,
        *,
        purpose: str,
        query: str = "",
    ) -> None:
        fingerprint = self._catalog_fingerprint(api_key)
        cached = self._cached_catalog(provider, api_key)
        if cached is not None:
            if purpose == "key":
                self._complete_key_validation(provider, api_key, fingerprint)
            elif purpose == "models":
                self._emit_catalog(provider, cached, query)
            return

        request_key = (provider.key, fingerprint)
        with self._catalog_lock:
            if request_key in self._catalog_inflight:
                return
            self._catalog_inflight.add(request_key)
            thread = threading.Thread(
                target=self._catalog_worker,
                args=(provider, api_key, fingerprint, purpose, query),
                daemon=True,
                name=f"soca-catalog-{provider.key}",
            )
            self._catalog_threads.add(thread)
        thread.start()

    def _catalog_worker(
        self,
        provider: LLMProvider,
        api_key: str,
        fingerprint: str,
        purpose: str,
        query: str,
    ) -> None:
        try:
            catalog = self.catalog_fetcher(provider, api_key)
        except RemoteLLMError as exc:
            if purpose == "key":
                if self._key_validation_is_current(provider.key, fingerprint):
                    self._emit_key_status(provider.key, ok=False, message=str(exc))
            else:
                self._error(str(exc), provider=provider.key)
            return
        except Exception as exc:  # noqa: BLE001 - external catalog adapters are untrusted
            LOGGER.warning(
                "Unexpected model catalog failure for provider %s (%s)",
                provider.key,
                type(exc).__name__,
            )
            if purpose == "key":
                if self._key_validation_is_current(provider.key, fingerprint):
                    self._emit_key_status(
                        provider.key,
                        ok=False,
                        message="Không thể xác thực API key. Hãy thử lại sau.",
                    )
            else:
                self._error(
                    f"Không thể lấy danh sách model của {provider.label}.",
                    provider=provider.key,
                )
            return
        finally:
            request_key = (provider.key, fingerprint)
            with self._catalog_lock:
                self._catalog_inflight.discard(request_key)
                self._catalog_threads.discard(threading.current_thread())

        with self._catalog_lock:
            self._catalog_cache[provider.key] = (fingerprint, list(catalog))

        self._refresh_active_model_capabilities(provider, catalog)

        if purpose == "key":
            self._complete_key_validation(provider, api_key, fingerprint)
        elif purpose in {"models", "config"}:
            self._emit_catalog(provider, catalog, query)

        # Startup can emit llm_config before this fetch is ready. Refresh only
        # its pricing field once the cache is available.
        if purpose != "key" and (
            self.llm_settings.backend == "remote" and self.llm_settings.provider_key == provider.key
        ):
            self._emit_llm_config()

    def _refresh_active_model_capabilities(
        self,
        provider: LLMProvider,
        catalog: list[RemoteModelInfo],
    ) -> None:
        settings = self.llm_settings
        if settings.backend != "remote" or settings.provider_key != provider.key:
            return
        model = next((item for item in catalog if item.id == settings.model_id), None)
        if model is None:
            return
        refreshed = settings.with_model_capabilities(
            context_window=model.context_length,
            max_output_tokens=model.max_output_tokens,
            reasoning_supported=model.reasoning_supported,
            reasoning_mandatory=model.reasoning_mandatory,
            reasoning_parameter=model.reasoning_parameter,
        )
        if refreshed == settings:
            return
        if not self._dispose_text_bundle():
            LOGGER.error("Refusing to refresh model capabilities while text runtime is open")
            return
        try:
            self.llm_settings_saver(refreshed)
        except ValueError:
            LOGGER.warning("Could not persist refreshed model capabilities", exc_info=True)
            return
        self.llm_settings = refreshed
        self._invalidate_voice_runtime()
        self._last_prompt_manifest = None

    def _key_validation_is_current(self, provider_key: str, fingerprint: str) -> bool:
        with self._catalog_lock:
            return self._key_validation_tokens.get(provider_key) == fingerprint

    def _complete_key_validation(
        self,
        provider: LLMProvider,
        api_key: str,
        fingerprint: str,
    ) -> None:
        if not self._key_validation_is_current(provider.key, fingerprint):
            return
        if self._voice_is_active():
            self._emit_key_status(
                provider.key,
                ok=False,
                message="Không thể đổi API key khi voice đang chạy; hãy dừng voice trước.",
            )
            return
        if not self._dispose_text_bundle():
            self._emit_key_status(
                provider.key,
                ok=False,
                message="Không thể đóng text runtime hiện tại để đổi API key.",
            )
            return
        self.secret_store.set_key(provider.key, api_key)
        self._invalidate_voice_runtime()
        self._emit_key_status(
            provider.key,
            ok=True,
            masked=self.secret_store.mask(api_key),
        )

    def _provider_from_command(self, command: dict[str, Any]) -> LLMProvider | None:
        raw_provider = command.get("provider")
        if not isinstance(raw_provider, str):
            self._error("LLM provider phải là chuỗi.")
            return None
        try:
            return get_provider(raw_provider)
        except ValueError as exc:
            self._error(str(exc))
            return None

    def _emit_key_status(
        self,
        provider_key: str,
        *,
        ok: bool,
        masked: str | None = None,
        message: str | None = None,
        pending: bool = False,
    ) -> None:
        payload: dict[str, Any] = {
            "event": "llm_key_status",
            "provider": provider_key,
            "ok": ok,
        }
        if pending:
            payload["pending"] = True
        if masked is not None:
            payload["masked"] = masked
        if message is not None:
            payload["message"] = message
        self.writer.emit(payload)

    # --- memory & usage -----------------------------------------------------------

    def _cmd_memory(self) -> None:
        if self.session_memory is None:
            self.writer.emit(
                {
                    "event": "memory",
                    "enabled": False,
                    "text": "",
                    "summary": "",
                    "recent": "",
                    "stats": None,
                }
            )
            return
        summary, recent = self.session_memory.working.render_sections()
        rendered = self.session_memory.render().strip()
        self.writer.emit(
            {
                "event": "memory",
                "enabled": True,
                "text": rendered,
                "summary": summary,
                "recent": recent,
                "stats": dataclasses.asdict(self.session_memory.stats()),
            }
        )

    def _cmd_context(self) -> None:
        stats = self.session_memory.stats() if self.session_memory is not None else None
        manifest = self._last_prompt_manifest
        if isinstance(manifest, dict):
            self._emit_context_manifest(manifest, stats=stats, estimated=False)
            return

        try:
            manifest = self._build_resident_context_manifest()
        except PromptBudgetError as exc:
            self.writer.emit(
                {
                    "event": "context",
                    "estimated": True,
                    "ready": False,
                    "context_error": exc.code,
                    "context_error_detail": exc.detail,
                    "token_counter": Utf8TokenCounter.name,
                    "session": dataclasses.asdict(stats) if stats is not None else None,
                    "model_context_tokens": self._model_context_length(),
                    "output_reserve_tokens": self.llm_settings.effective_max_tokens,
                    "components": [],
                }
            )
            return

        self._emit_context_manifest(manifest, stats=stats, estimated=True)

    def _context_component_rows(
        self,
        manifest: dict[str, Any],
        *,
        include_dynamic: bool = True,
    ) -> list[dict[str, Any]]:
        labels = {
            "system": "System instructions",
            "current_input": "Current user input",
            "answer_prefix": "Prompt scaffolding",
            "joint_grounding_policy": "Grounding policy",
            "answer_policy": "Answer policy",
            "memory": "Memory context",
            "core_memory": "Core memory",
            "working_summary": "Working summary",
            "recent_conversation": "Recent conversation",
            "knowledge": "Knowledge retrieval",
            "archive_memory": "Archive memory retrieval",
            "prompt_scaffolding": "Prompt scaffolding",
        }
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        raw_components = manifest.get("components")
        if isinstance(raw_components, list):
            for item in raw_components:
                if not isinstance(item, dict):
                    continue
                component_id = str(item.get("component_id") or "unknown")
                seen.add(component_id)
                required = bool(item.get("required"))
                rows.append(
                    {
                        "id": component_id,
                        "label": labels.get(component_id, component_id),
                        "tokens": item.get("tokens"),
                        "included": bool(item.get("included")),
                        "required": required,
                        "priority": item.get("priority"),
                        "policy": "always" if required else "on_demand",
                    }
                )
        if include_dynamic:
            for component_id, label, policy in (
                ("archive_memory", "Archive memory retrieval", "on_demand"),
                ("knowledge", "Knowledge retrieval", "on_demand"),
                ("current_input", "Current user input", "per_turn"),
            ):
                if component_id in seen:
                    continue
                rows.append(
                    {
                        "id": component_id,
                        "label": label,
                        "tokens": None,
                        "included": False,
                        "required": False,
                        "priority": None,
                        "policy": policy,
                    }
                )
        return rows

    def _emit_context_manifest(
        self,
        manifest: dict[str, Any],
        *,
        stats: Any,
        estimated: bool,
    ) -> None:
        prompt_tokens = manifest.get("prompt_tokens")
        input_budget = manifest.get("input_budget_tokens")
        available = (
            max(0, input_budget - prompt_tokens)
            if isinstance(input_budget, int) and isinstance(prompt_tokens, int)
            else None
        )
        payload: dict[str, Any] = {
            "event": "context",
            "estimated": estimated,
            "ready": True,
            "token_counter": manifest.get("token_counter"),
            "prompt_hash": manifest.get("prompt_hash"),
            "prompt_manifest": manifest,
            "session": dataclasses.asdict(stats) if stats is not None else None,
            "resident_prompt_tokens": prompt_tokens,
            "output_reserve_tokens": manifest.get("effective_output_tokens"),
            "model_context_tokens": manifest.get("context_window"),
            "input_budget_tokens": input_budget,
            "available_dynamic_tokens": available,
            "observed_prompt_tokens": manifest.get("observed_prompt_tokens"),
            "observed_prompt_token_source": manifest.get("observed_prompt_token_source"),
            "provider_prompt_tokens": manifest.get("provider_prompt_tokens"),
            "prompt_token_delta": manifest.get("prompt_token_delta"),
            "components": self._context_component_rows(manifest),
        }
        self.writer.emit(payload)

    def _build_resident_context_manifest(self) -> dict[str, Any]:
        core_memory = self._core_memory_text()
        core_section = f"Long-term memory:\n{core_memory}" if core_memory else ""
        summary_section = ""
        recent_section = ""
        if self.session_memory is not None:
            summary_section, recent_section = self.session_memory.working.render_sections()
        components = [
            PromptComponent(
                "system",
                SOCA_RUNTIME_SYSTEM_PROMPT.strip(),
                priority=0,
                required=True,
            ),
            PromptComponent("core_memory", core_section, priority=30),
            PromptComponent("working_summary", summary_section, priority=30),
            PromptComponent("recent_conversation", recent_section, priority=30),
            PromptComponent("answer_prefix", "Trả lời cuối cùng:", priority=0, required=True),
        ]
        model_id = self.text_config.llm_model if self.text_config.llm_model_is_override else self.llm_settings.model_id
        source = "remote_catalog" if self.llm_settings.backend == "remote" else "local_registry"
        capability = capability_from_values(
            model_id=model_id,
            context_window=self._model_context_length(),
            max_output_tokens=self.llm_settings.model_max_output_tokens,
            tokenizer=Utf8TokenCounter.name,
            source=source,
        )
        _, manifest = PromptAssembler(
            capability,
            counter=Utf8TokenCounter(),
        ).assemble(
            components,
            requested_output_tokens=self.llm_settings.effective_max_tokens,
        )
        return manifest.to_dict()

    def _core_memory_text(self) -> str:
        vault = self.text_config.vault
        if not vault.is_dir():
            return ""
        try:
            return CoreMemoryStore(
                vault,
                max_chars=self.text_config.memory_item_chars,
            ).read_core()
        except (OSError, UnicodeError, ValueError):
            return ""

    def _model_context_length(self) -> int | None:
        if self.text_config.llm_model_is_override:
            config = LLM_MODEL_REGISTRY.get(self.text_config.llm_model)
            return config.context_window if config is not None else None
        settings = self.llm_settings
        if settings.backend == "local":
            config = LLM_MODEL_REGISTRY.get(settings.model_id)
            return config.context_window if config is not None else None
        active = self._active_remote_model(settings)
        return active.context_length if active is not None else None

    def _cmd_memory_compact(self, action: str) -> None:
        if self.session_memory is None:
            self.writer.emit(
                {"event": "memory_compaction", "status": "disabled", "detail": "memory disabled"}
            )
            return
        if action == "status":
            result = self.session_memory.compaction_status()
        elif action == "cancel":
            result = self.session_memory.cancel_compaction()
        elif action == "request":
            result = self.session_memory.request_compaction()
        else:
            self._error("memory compact action must be request, status, or cancel")
            return
        self.writer.emit({"event": "memory_compaction", **dataclasses.asdict(result)})
        if result.status == "accepted":
            self._cmd_context()
        elif result.status not in {"running", "idle"}:
            self._cmd_memory()
            self._cmd_context()

    def _memory_commands(self) -> MemoryCommands | None:
        vault = self.text_config.vault
        if not vault.is_dir():
            return None
        return MemoryCommands(vault, ProposalStore(vault / "memory" / ".proposals"))

    def _cmd_memory_proposals(self) -> None:
        commands = self._memory_commands()
        proposals = commands.list_pending() if commands is not None else ()
        self.writer.emit(
            {
                "event": "memory_proposals",
                "proposals": [
                    {
                        "id": proposal.id,
                        "kind": proposal.kind,
                        "statement": proposal.statement[:400],
                        "confidence": proposal.confidence,
                        "createdAt": proposal.created_at.isoformat(),
                    }
                    for proposal in proposals[:64]
                ],
            }
        )

    def _cmd_memory_action(self, command: dict[str, Any], action: str) -> None:
        proposal_id = command.get("proposal_id")
        if not isinstance(proposal_id, str):
            self._error("proposal_id must be a string")
            return
        commands = self._memory_commands()
        if commands is None:
            self.writer.emit(
                {
                    "event": "memory_action",
                    "proposal_id": proposal_id[:80],
                    "action": "approved" if action == "approve" else "rejected",
                    "ok": False,
                    "error_code": "memory_unavailable",
                }
            )
            return
        try:
            result = getattr(commands, action)(proposal_id)
            self.writer.emit(
                {
                    "event": "memory_action",
                    "proposal_id": proposal_id,
                    "action": "approved" if action == "approve" else "rejected",
                    "ok": result.status in {"approved", "rejected"},
                    "error_code": None
                    if result.status in {"approved", "rejected"}
                    else result.status,
                }
            )
        except (OSError, ValueError, KeyError) as exc:
            LOGGER.warning("Memory command failed (%s)", type(exc).__name__)
            self.writer.emit(
                {
                    "event": "memory_action",
                    "proposal_id": proposal_id[:80],
                    "action": "approved" if action == "approve" else "rejected",
                    "ok": False,
                    "error_code": "command_failed",
                }
            )

    def _cmd_usage(self) -> None:
        with self._usage_lock:
            usage = self.session_usage
        self.writer.emit(
            {
                "event": "usage",
                "turns": usage.total_turns,
                "llm_turns": usage.llm_turns,
                "prompt_tokens": usage.total_prompt_tokens,
                "completion_tokens": usage.total_completion_tokens,
                "mean_ttft_ms": usage.mean_ttft_ms,
                "mean_tokens_per_second": usage.mean_tokens_per_second,
            }
        )

    def _track_usage(self, usage: TurnUsage | None) -> None:
        if usage is None:
            return
        with self._usage_lock:
            self.session_usage = self.session_usage.add(usage)

    def _emit_turn_progress(
        self,
        surface: str,
        phase: str,
        *,
        operation: str = "",
        status: str = "active",
        context: _TurnProgressContext | None = None,
        terminal_status: str | None = None,
        detail: str | None = None,
    ) -> None:
        selected = context
        if selected is None:
            with self._progress_lock:
                selected = self._progress_contexts.get(surface)
        if selected is None:
            return
        with self._progress_lock:
            sequence = selected.sequence
            selected.sequence += 1
        payload: dict[str, Any] = {
            "event": "turn_progress",
            "surface": surface,
            "phase": phase,
            "operation": operation,
            "status": status,
            "run_id": selected.run_id,
            "goal_id": selected.goal_id,
            "sequence": sequence,
        }
        if terminal_status is not None:
            payload["terminal_status"] = terminal_status
        if detail:
            payload["detail"] = detail
        self.writer.emit(payload)

    def _start_turn_progress(self, surface: str) -> _TurnProgressContext:
        run_id = uuid4().hex
        active_goal = self.active_goal_store.current
        goal_id = active_goal.goal_id if active_goal is not None else f"pending-{run_id}"
        workflow_surface = "voice" if surface == "voice" else "chat"
        context = _TurnProgressContext(
            surface,
            run_id,
            goal_id,
            WorkflowEventStream(
                session_id=self.text_config.session_id,
                run_id=run_id,
                goal_id=goal_id,
                surface=workflow_surface,
            ),
        )
        with self._progress_lock:
            self._progress_contexts[surface] = context
        self._emit_workflow_event(context, EventType.TURN_STARTED, TurnNode.ADMIT, EventStatus.STARTED)
        return context

    def _clear_turn_progress(self, context: _TurnProgressContext) -> None:
        with self._progress_lock:
            if self._progress_contexts.get(context.surface) is context:
                self._progress_contexts.pop(context.surface, None)

    def _emit_workflow_event(
        self,
        context: _TurnProgressContext,
        event: EventType,
        node: TurnNode,
        status: EventStatus = EventStatus.ACTIVE,
        payload: dict[str, Any] | None = None,
    ) -> None:
        item = context.workflow.emit(event, node, status=status, payload=payload)
        self.writer.emit(workflow_event_to_protocol(item))

    def _emit_workflow_for_result(self, result: Any, context: _TurnProgressContext) -> None:
        self._emit_workflow_event(
            context,
            EventType.STEP_COMPLETED,
            TurnNode.SYNTHESIZE,
            EventStatus.COMPLETED,
            {"route": result.route.value},
        )
        if result.response_text.strip():
            self._emit_workflow_event(
                context,
                EventType.ANSWER_DELTA,
                TurnNode.SYNTHESIZE,
                payload={"text": result.response_text},
            )
        terminal = context.workflow.emit_terminal(terminal_from_runtime_result(result))
        self.writer.emit(workflow_event_to_protocol(terminal))

    def _emit_runtime_progress(self, surface: str, stage: str) -> None:
        phase = _progress_phase_for_stage(stage)
        self._emit_turn_progress(
            surface,
            phase,
            operation=stage,
        )
        with self._progress_lock:
            context = self._progress_contexts.get(surface)
        if context is not None:
            self._emit_workflow_event(
                context,
                EventType.STEP_PROGRESS,
                _workflow_node_for_phase(phase),
                payload={"operation": stage},
            )

    # --- chat -------------------------------------------------------------------

    def _cmd_chat(self, text: str) -> None:
        if not text.strip():
            self._error("chat text is empty")
            return
        if self.no_model:
            self._error("chat unavailable: engine started with --no-model")
            return
        if not self._chat_lock.acquire(blocking=False):
            self._error("chat busy: previous turn still running")
            return
        thread = threading.Thread(
            target=self._chat_worker, args=(text,), daemon=True, name="soca-engine-chat"
        )
        self._chat_thread = thread
        thread.start()

    def _chat_worker(self, text: str) -> None:
        terminal_emitted = False
        progress = self._start_turn_progress("chat")
        try:
            self.writer.emit(
                {
                    "event": "chat",
                    "type": "start",
                    "text": text,
                    "run_id": progress.run_id,
                    "goal_id": progress.goal_id,
                }
            )
            self._emit_turn_progress(
                "chat",
                "preparing",
                operation="runtime",
                context=progress,
            )
            bundle = self._ensure_text_bundle()
            normalized_text, metadata = normalize_text_turn(text)
            progress_setter = getattr(bundle.runtime, "set_progress_callback", None)
            if callable(progress_setter):
                progress_setter(lambda stage: self._emit_runtime_progress("chat", str(stage)))
            self._emit_turn_progress(
                "chat",
                "analyzing",
                operation="normalize_input",
                context=progress,
            )
            try:
                result = bundle.runtime.run_text_turn(
                    normalized_text, source="engine_chat", metadata=metadata
                )
            finally:
                if callable(progress_setter):
                    progress_setter(None)
            usage = TurnUsage.from_runtime_result(result)
            self._track_usage(usage)
            self._emit_workflow_for_result(result, progress)
            self.writer.emit(
                {
                    "event": "chat",
                    "type": "done",
                    "text": answer_text_without_citation_labels(
                        result.response_text,
                        result.citations,
                    ),
                    "route": result.route.value,
                    "blocked": result.blocked,
                    "usage": usage,
                    "citations": list(citation_records(result.citations)),
                    "provider_trace": (
                        dict(result.trace.provider_trace) if result.trace is not None else {}
                    ),
                    "llm_error": (
                        dict(result.trace.llm_error) if result.trace is not None else {}
                    ),
                }
            )
            terminal_emitted = True
            self._emit_turn_progress(
                "chat",
                "complete",
                status="done",
                context=progress,
                terminal_status=_terminal_status_for_result(result),
            )
            trace = result.trace
            if trace is not None:
                self._last_prompt_manifest = trace.prompt_manifest
                self.writer.emit(
                    {
                        "event": "router_trace",
                        "tier": trace.tool_router_tier,
                        "tool": trace.tool_calls[-1].name if trace.tool_calls else None,
                        "reason": trace.tool_router_reason,
                        "disposition": trace.disposition,
                        "handler": trace.router_handler,
                        "selected_routes": list(trace.selected_routes),
                        "sources": list(trace.selected_sources),
                        "scores": dict(trace.router_scores),
                        "source_scores": dict(trace.router_source_scores),
                        "runner_up": trace.router_runner_up,
                        "margin": trace.router_margin,
                        "evidence_status": trace.evidence_status,
                        "evidence_completion_status": trace.evidence_completion_status,
                        "evidence_completion_reason": trace.evidence_completion_reason,
                        "evidence_completion_actions": trace.evidence_completion_actions,
                        "answer_policy": trace.answer_policy,
                        "answer_policy_reason": trace.answer_policy_reason,
                        "grounding_policy_version": trace.grounding_policy_version,
                        "citation_count": trace.citation_count,
                        "memory_access_plan": (
                            dataclasses.asdict(trace.memory_access_plan)
                            if trace.memory_access_plan is not None
                            else None
                        ),
                        "latency_ms": trace.stage_latencies_ms.get("tool_router", 0.0),
                    }
                )
                knowledge_hits = trace.knowledge_hits[:16]
                has_knowledge_decision = any(
                    getattr(item, "source", None) == "knowledge"
                    for item in trace.evidence_decisions
                )
                if knowledge_hits or has_knowledge_decision:
                    self.writer.emit(_retrieval_trace_payload(result, trace, knowledge_hits))
                self.writer.emit(self._memory_trace_payload(trace, self._pending_proposal_count()))
            self._cmd_context()
        except Exception as exc:  # noqa: BLE001 - protocol boundary must not crash
            if not terminal_emitted:
                outcome = TerminalOutcome(
                    status=TerminalStatus.SYSTEM_FAILURE,
                    final_text="",
                    goal_status=GoalStatus.FAILED,
                    error_code="chat_exception",
                    metadata={
                        "surface": "chat",
                        "source": "engine",
                        "exception_type": type(exc).__name__,
                    },
                )
                terminal = progress.workflow.emit_terminal(outcome)
                self.writer.emit(workflow_event_to_protocol(terminal))
                self._emit_turn_progress(
                    "chat",
                    "complete",
                    operation="runtime_error",
                    status="failed",
                    context=progress,
                    terminal_status="system_failure",
                    detail=type(exc).__name__,
                )
                self.writer.emit({"event": "chat", "type": "error", "text": str(exc)})
            else:
                LOGGER.exception("chat worker failed after terminal result")
        finally:
            self._clear_turn_progress(progress)
            # Worker cleanup is not a product terminal. An exception before a
            # result therefore emits chat:error, but never chat completion.
            self._chat_lock.release()

    def _ensure_text_bundle(self) -> TextRuntimeBundle:
        if self.text_bundle is not None:
            return self.text_bundle
        if self._settings_error is not None:
            raise RuntimeError("llm_settings_invalid")
        self.writer.emit({"event": "chat", "type": "loading", "text": "building text runtime"})
        runtime_config = dataclasses.replace(
            self.text_config,
            max_tokens=self.llm_settings.effective_max_tokens,
            temperature=self.llm_settings.temperature,
            top_p=self.llm_settings.top_p,
        )
        bundle = self.text_runtime_builder(
            runtime_config,
            session_memory=self.session_memory,
            llm_settings=self.llm_settings,
            secret_store=self.secret_store,
            engine_factory=self.llm_engine_factory,
            active_goal_store=self.active_goal_store,
        )
        self.text_bundle = bundle
        self.writer.emit(
            {
                "event": "chat",
                "type": "ready",
                "llm_status": bundle.llm_status,
                "knowledge_status": bundle.knowledge_status,
                "memory_status": bundle.memory_status,
            }
        )
        return bundle

    def _memory_trace_payload(self, trace: Any, pending_count: int | None) -> dict[str, Any]:
        stats = self.session_memory.stats() if self.session_memory is not None else None
        compaction = (
            self.session_memory.compaction_status() if self.session_memory is not None else None
        )
        compaction_status = getattr(compaction, "status", "idle")
        if compaction_status in {"accepted", "running"}:
            background_status = "queued" if compaction_status == "accepted" else "running"
        elif compaction_status in {"failed", "unavailable"}:
            background_status = "failed"
        else:
            background_status = "idle"
        compacted_turn_count = (
            getattr(compaction, "compacted_turns", None)
            if compaction_status in {"published", "trim_only", "stale"}
            else None
        )
        return {
            "event": "memory_trace",
            "mode": _memory_protocol_mode(
                trace.get("memory_mode", "none")
                if isinstance(trace, dict)
                else getattr(trace, "memory_mode", "none"),
                trace.get("memory_degraded_reason", "")
                if isinstance(trace, dict)
                else getattr(trace, "memory_degraded_reason", ""),
                int(trace.get("memory_hit_count", 0))
                if isinstance(trace, dict)
                else len(getattr(trace, "memory_hits", ())),
            ),
            "degraded_reason": (
                trace.get("memory_degraded_reason", "")
                if isinstance(trace, dict)
                else getattr(trace, "memory_degraded_reason", "")
            ),
            "hits": [
                {
                    "id": str(getattr(hit.document, "id", ""))[:120],
                    "corpus": (
                        "episode"
                        if str(getattr(hit.document, "path", "")).startswith("memory/episodes/")
                        or getattr(hit.document, "retrieval_backend", "") == "memory_episode"
                        else "profile"
                    ),
                    "relevance": float(
                        getattr(getattr(hit, "score", None), "relevance", 0.0)
                    ),
                    "recency": float(getattr(getattr(hit, "score", None), "recency", 0.0)),
                    "importance": float(
                        getattr(getattr(hit, "score", None), "importance", 0.0)
                    ),
                    "total": float(getattr(getattr(hit, "score", None), "total", 0.0)),
                }
                for hit in (
                    getattr(trace, "memory_hits", ())[:16]
                    if not isinstance(trace, dict)
                    else ()
                )
            ],
            "hit_count": (
                int(trace.get("memory_hit_count", 0))
                if isinstance(trace, dict)
                else len(getattr(trace, "memory_hits", ()))
            ),
            "compacted_turn_count": compacted_turn_count,
            "recent_turn_count": stats.turn_count if stats is not None else None,
            "background_status": background_status,
            "summary_worker_state": stats.worker_state if stats is not None else "disabled",
            "summary_generation": stats.summary_generation if stats is not None else None,
            "pending_compaction": stats.pending_compaction if stats is not None else False,
            "pending_proposal_count": pending_count,
        }

    def _create_session_memory(self) -> SessionMemory | None:
        config = self.text_config
        if config.no_memory:
            return None
        return SessionMemory(
            thread_id=config.session_id,
            max_turns=config.session_turns,
            max_chars=config.session_chars,
            max_turn_chars=config.turn_chars,
            summary_enabled=not self.no_model,
            summary_threads=config.llm_threads,
            summary_gpu_layers=config.llm_gpu_layers,
            persistence=config.session_persistence,
            checkpoint_store=(
                SessionCheckpointStore(default_session_checkpoint_home())
                if config.session_persistence == "local_resumable"
                else None
            ),
            resume=config.session_resume,
        )

    # --- voice ------------------------------------------------------------------

    def _cmd_voice_profile_select(self, command: dict[str, Any]) -> None:
        """Apply one ready voice profile without rebuilding session memory."""
        if self.voice_config is None:
            self._error(
                "voice profile unavailable: no voice runtime config",
                code="voice_profile_unavailable",
            )
            return
        if self._voice_is_active():
            self._error(
                "Không thể đổi ASR khi voice đang chạy; hãy dừng voice trước.",
                code="voice_profile_change_while_active",
            )
            return
        profile_key = command.get("profile")
        if not isinstance(profile_key, str) or not profile_key:
            self._error("voice profile phải là chuỗi.", code="voice_profile_invalid")
            return
        try:
            profile = get_voice_runtime_profile(profile_key)
            from soca.app.profiles import asr_readiness

            readiness = asr_readiness(profile.asr)
        except (KeyError, ValueError, RuntimeError) as exc:
            self._error(
                "voice profile không hợp lệ",
                code="voice_profile_invalid",
                detail=str(exc),
            )
            return
        if not readiness.ok:
            self._error(
                "voice profile chưa sẵn sàng",
                code="voice_profile_not_ready",
                profile=profile_key,
                asr=profile.asr_model,
                readiness=readiness.status,
                detail=readiness.detail,
            )
            return
        if not self._invalidate_voice_runtime():
            return
        self.voice_config = dataclasses.replace(
            self.voice_config,
            profile_key=profile_key,
            asr=profile.asr,
        )
        self.profile = profile_key
        self.hello()
        self._cmd_status()

    def _cmd_voice_start(self, max_turns: int | None) -> None:
        if self._settings_error is not None:
            self._error(self._settings_error, code="llm_settings_invalid")
            return
        if self.no_model:
            self._error("voice unavailable: engine started with --no-model")
            return
        if self.voice_config is None:
            self._error("voice unavailable: no voice runtime config")
            return
        if self.voice_stop_event is not None and not self.voice_stop_event.is_set():
            self._error("voice already running")
            return

        controller = self._ensure_voice_controller()
        stop_event = threading.Event()
        self.voice_stop_event = stop_event
        queue: Queue[VoiceMonitorEvent | None] = Queue()

        pump = threading.Thread(
            target=self._voice_pump, args=(queue,), daemon=True, name="soca-engine-voice-pump"
        )
        worker = threading.Thread(
            target=controller.run_loop,
            args=(queue,),
            kwargs={"stop_event": stop_event, "max_turns": max_turns},
            daemon=True,
            name="soca-engine-voice-loop",
        )
        self._voice_threads = [pump, worker]
        pump.start()
        worker.start()

    def _voice_pump(self, queue: Queue[VoiceMonitorEvent | None]) -> None:
        while True:
            event = queue.get()
            if event is None:
                break
            self.writer.emit(
                {
                    "event": "voice",
                    "type": event.type,
                    "text": event.text,
                    "latency_ms": event.latency_ms,
                    "metadata": event.metadata,
                    "usage": event.usage,
                }
            )
            if event.type == "turn_start":
                progress = self._start_turn_progress("voice")
                self._emit_turn_progress(
                    "voice",
                    "preparing",
                    operation="voice_turn",
                    context=progress,
                )
            elif event.type == "runtime":
                metadata = event.metadata
                prompt_manifest = metadata.get("prompt_manifest")
                self._last_prompt_manifest = (
                    prompt_manifest if isinstance(prompt_manifest, dict) else None
                )
                tier = metadata.get("router_tier", "none")
                if tier not in {"deterministic", "semantic", "llm", "none"}:
                    tier = "none"
                self.writer.emit(
                    {
                        "event": "router_trace",
                        "tier": tier,
                        "tool": None,
                        "reason": metadata.get("router_reason", "no_match"),
                        "disposition": metadata.get("router_disposition", "unresolved"),
                        "handler": metadata.get("router_handler"),
                        "selected_routes": metadata.get("router_selected_routes", []),
                        "sources": metadata.get("router_sources", []),
                        "scores": metadata.get("router_scores", {}),
                        "source_scores": metadata.get("router_source_scores", {}),
                        "runner_up": metadata.get("router_runner_up"),
                        "margin": metadata.get("router_margin"),
                        "evidence_status": metadata.get(
                            "evidence_status",
                            "not_requested",
                        ),
                        "answer_policy": metadata.get("answer_policy", "free_chat"),
                        "answer_policy_reason": metadata.get(
                            "answer_policy_reason",
                            "no_retrieval_evidence",
                        ),
                        "grounding_policy_version": metadata.get(
                            "grounding_policy_version",
                            "",
                        ),
                        "citation_count": int(metadata.get("citation_count", 0)),
                        "memory_access_plan": metadata.get("memory_access_plan"),
                        "latency_ms": float(metadata.get("router_latency_ms", 0.0)),
                    }
                )
                self.writer.emit(self._memory_trace_payload(metadata, self._pending_proposal_count()))
            elif event.type == "progress":
                stage = str(event.metadata.get("stage") or "")
                self._emit_runtime_progress("voice", stage)
            elif event.type == "llm_token":
                with self._progress_lock:
                    progress = self._progress_contexts.get("voice")
                if progress is not None and event.text:
                    self._emit_workflow_event(
                        progress,
                        EventType.ANSWER_DELTA,
                        TurnNode.SYNTHESIZE,
                        payload={"text": event.text},
                    )
            elif event.type == "recording":
                self._emit_turn_progress(
                    "voice",
                    "preparing",
                    operation="listening",
                )
            elif event.type == "transcribing":
                self._emit_turn_progress(
                    "voice",
                    "analyzing",
                    operation="speech_recognition",
                )
            elif event.type in {"tts", "playback_started", "audio"}:
                self._emit_turn_progress(
                    "voice",
                    "speech",
                    operation="text_to_speech",
                )
            self._track_usage(event.usage)
            if event.type == "done":
                with self._progress_lock:
                    progress = self._progress_contexts.get("voice")
                if progress is not None:
                    terminal_status = str(event.metadata.get("terminal_status") or "system_failure")
                    self._emit_voice_workflow_terminal(progress, event, terminal_status)
                    progress_status = (
                        "cancelled"
                        if terminal_status == "cancelled"
                        else "failed"
                        if terminal_status in {"safe_failure", "budget_exhausted", "system_failure"}
                        else "done"
                    )
                    self._emit_turn_progress(
                        "voice",
                        "complete",
                        status=progress_status,
                        context=progress,
                        terminal_status=terminal_status,
                    )
                    self._clear_turn_progress(progress)
                self._cmd_context()
            elif event.type == "error":
                with self._progress_lock:
                    progress = self._progress_contexts.get("voice")
                if progress is not None:
                    self._emit_voice_workflow_terminal(progress, event, "system_failure")
                    self._emit_turn_progress(
                        "voice",
                        "complete",
                        operation="runtime_error",
                        status="failed",
                        context=progress,
                        terminal_status="system_failure",
                        detail=type(event.text).__name__,
                    )
                    self._clear_turn_progress(progress)
        if self.voice_stop_event is not None:
            self.voice_stop_event.set()

    def _emit_voice_workflow_terminal(
        self,
        context: _TurnProgressContext,
        event: VoiceMonitorEvent,
        terminal_status: str,
    ) -> None:
        if context.workflow.terminal_outcome is not None:
            return
        try:
            status = TerminalStatus(terminal_status)
        except ValueError:
            status = TerminalStatus.SYSTEM_FAILURE
            terminal_status = status.value
        goal_status = (
            GoalStatus.WAITING_FOR_USER
            if status is TerminalStatus.NEEDS_CLARIFICATION
            else GoalStatus.ACHIEVED
            if status is TerminalStatus.ACHIEVED
            else GoalStatus.FAILED
        )
        outcome = TerminalOutcome(
            status=status,
            final_text=event.text,
            goal_status=goal_status,
            recoverable=status is TerminalStatus.NEEDS_CLARIFICATION,
            route=str(event.metadata.get("runtime_route") or ""),
            error_code=None if status is TerminalStatus.ACHIEVED else terminal_status,
            metadata={"surface": "voice", "source": "pipeline"},
        )
        terminal = context.workflow.emit_terminal(outcome)
        self.writer.emit(workflow_event_to_protocol(terminal))

    def _pending_proposal_count(self) -> int | None:
        commands = self._memory_commands()
        if commands is None:
            return 0
        try:
            return len(commands.list_pending())
        except (OSError, ValueError):
            LOGGER.warning("proposal telemetry unavailable", exc_info=True)
            return None

    def _cmd_voice_stop(self) -> bool:
        cleanup_failed = False
        if self.voice_stop_event is not None:
            self.voice_stop_event.set()
        if self.voice_controller is not None:
            try:
                self.voice_controller.stop()
            except Exception as exc:  # noqa: BLE001 - surface teardown failure in protocol
                cleanup_failed = True
                self._error(
                    "voice runtime cleanup failed",
                    code="voice_runtime_cleanup_failed",
                    detail=type(exc).__name__,
                )
            else:
                self.voice_controller = None
        return cleanup_failed

    def _voice_is_active(self) -> bool:
        stop_event = self.voice_stop_event
        return bool(
            stop_event is not None
            and not stop_event.is_set()
            and any(thread.is_alive() for thread in self._voice_threads)
        )

    def _invalidate_voice_runtime(self) -> bool:
        """Drop an idle voice bundle so the next start sees current settings."""
        if self._voice_is_active():
            # A background catalog refresh may race with a voice start.  Keep
            # the active bundle unchanged; the command-level mutation guard
            # prevents intentional hot-swaps and the next idle refresh will
            # rebuild from the persisted settings.
            return True
        controller = self.voice_controller
        if controller is not None:
            try:
                controller.stop()
            except Exception as exc:  # noqa: BLE001 - lifecycle boundary
                self._error(
                    "voice runtime cleanup failed",
                    code="voice_runtime_cleanup_failed",
                    detail=type(exc).__name__,
                )
                return False
        self.voice_controller = None
        return True

    def _selected_voice_settings(self) -> LlmSettings:
        settings = self.llm_settings
        if self.voice_config is not None and self.voice_config.llm_model_is_override:
            return settings.with_backend("local").with_model(self.voice_config.llm_model)
        return settings

    def _ensure_voice_controller(self) -> VoiceMonitorController:
        if self.voice_controller is not None:
            return self.voice_controller
        assert self.voice_config is not None

        kwargs: dict[str, Any] = {}
        if self.voice_runtime_builder is not None:
            kwargs["runtime_builder"] = self.voice_runtime_builder
        else:
            # The engine owns the selected settings and secret reader.  Pass
            # those exact objects into voice instead of relying on a second
            # disk read that could lag behind the chat selection.
            def build_selected_voice_runtime(
                config: ResolvedVoiceRuntimeConfig,
                *,
                session_memory: SessionMemory | None = None,
            ):
                return build_voice_runtime(
                    config,
                    session_memory=session_memory,
                    llm_settings=self._selected_voice_settings(),
                    secret_store=self.secret_store,
                    engine_factory=self.llm_engine_factory,
                    active_goal_store=self.active_goal_store,
                )

            kwargs["runtime_builder"] = build_selected_voice_runtime
        if self.voice_recorder is not None:
            kwargs["recorder"] = self.voice_recorder
        player = self.voice_player
        if player is None and not self.no_model:
            # Same Path B lazy build as the TUI: barge-in lives in the duplex sink.
            from soca.core.duplex_aec_sink import DuplexAecSink

            player = DuplexAecSink()
        if player is not None:
            kwargs["player"] = player

        self.voice_controller = VoiceMonitorController(
            self.voice_config,
            warmup=self.warmup_voice,
            session_memory=self.session_memory,
            **kwargs,
        )
        return self.voice_controller


def run_engine(
    *,
    voice_config: ResolvedVoiceRuntimeConfig | None,
    text_config: TextRuntimeConfig,
    profile: str,
    no_model: bool = False,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    text_runtime_builder: TextRuntimeBuilder = build_text_runtime,
    voice_runtime_builder: VoiceRuntimeBuilder | None = None,
    voice_recorder: VoiceRecorder | None = None,
    voice_player: AudioSink | None = None,
    warmup_voice: bool = True,
    llm_settings_loader: SettingsLoader = load_settings,
    llm_settings_saver: SettingsSaver = save_settings,
    secret_store: LlmSecretStore | None = None,
    catalog_fetcher: CatalogFetcher = fetch_catalog,
    llm_engine_factory: EngineBuilder = DEFAULT_LLM_ENGINE_FACTORY,
) -> int:
    """Run the engine loop until ``quit`` or EOF. Returns a process exit code."""
    reader = stdin or sys.stdin
    writer = _ProtocolWriter(stdout or sys.stdout)

    engine = SocaEngine(
        voice_config=voice_config,
        text_config=text_config,
        profile=profile,
        no_model=no_model,
        writer=writer,
        text_runtime_builder=text_runtime_builder,
        voice_runtime_builder=voice_runtime_builder,
        voice_recorder=voice_recorder,
        voice_player=voice_player,
        warmup_voice=warmup_voice,
        llm_settings_loader=llm_settings_loader,
        llm_settings_saver=llm_settings_saver,
        secret_store=secret_store,
        catalog_fetcher=catalog_fetcher,
        llm_engine_factory=llm_engine_factory,
    )

    # Keep protocol stdout pristine: reroute stray prints (model loaders,
    # warnings) to stderr for the duration of the loop.
    with contextlib.redirect_stdout(sys.stderr):
        engine.hello()
        try:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                try:
                    command = json.loads(line)
                except json.JSONDecodeError as exc:
                    engine._error(f"invalid JSON: {exc}", line=line[:200])
                    continue
                if not isinstance(command, dict):
                    engine._error("command must be a JSON object", line=line[:200])
                    continue
                if not engine.dispatch(command):
                    break
        finally:
            engine.shutdown()
    return 0


__all__ = ["PROTOCOL_VERSION", "SocaEngine", "run_engine"]
