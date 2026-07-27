"""Headless SoCa engine: NDJSON protocol over stdio for external UIs.

The Ink TUI (``ui/``) spawns ``soca engine`` and speaks this protocol:

* stdin  — one JSON command per line:
  ``{"cmd": "status"}`` · ``{"cmd": "chat", "text": "..."}`` ·
  ``{"cmd": "voice_start", "max_turns": null}`` · ``{"cmd": "voice_stop"}`` ·
  ``{"cmd": "llm_providers"}`` · ``{"cmd": "llm_models", "provider": "..."}`` ·
  ``{"cmd": "llm_set_key", "provider": "...", "key": "..."}`` ·
  ``{"cmd": "llm_select", "backend": "remote", "provider": "...", "model": "..."}`` ·
  ``{"cmd": "quit"}``
* stdout — one JSON event per line, first ``{"event": "hello", ...}`` then
  ``voice`` / ``chat`` / ``status`` / ``engine_error`` / ``bye`` events.

Audio never crosses the boundary: the mic, DuplexAecSink barge-in, and TTS
playback all stay inside this process. The UI only renders state.

stdout belongs to the protocol exclusively — everything else (model loading
chatter, warnings) is redirected to stderr while the engine runs.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from queue import Queue
from typing import Any, Protocol, TextIO

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
from soca.core.usage import SessionUsage, TurnUsage
from soca.llm.providers import (
    PRICING_TABLE_AS_OF,
    LLMProvider,
    RemoteLLMError,
    RemoteModelInfo,
    fetch_catalog,
    get_provider,
    search_models,
)
from soca.memory import MemoryCommands, ProposalStore, SessionMemory
from soca.tts import VALTEC_TTS_CONFIG

PROTOCOL_VERSION = 1
LOGGER = logging.getLogger(__name__)

TextRuntimeBuilder = Callable[..., TextRuntimeBundle]
SettingsLoader = Callable[[], LlmSettings]
SettingsSaver = Callable[[LlmSettings], None]
CatalogFetcher = Callable[[LLMProvider, str], list[RemoteModelInfo]]


def _memory_protocol_mode(
    mode: object,
    degraded_reason: object,
    hit_count: object,
) -> str:
    if degraded_reason or mode == "degraded":
        return "degraded"
    if mode == "retrieved" or (mode == "blob" and isinstance(hit_count, int) and hit_count > 0):
        return "retrieved"
    return "blob"


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
        self._settings_warning: str | None = None
        try:
            self.llm_settings = llm_settings_loader()
        except ValueError:
            LOGGER.warning(
                "Ignoring invalid persisted LLM settings; using local defaults", exc_info=True
            )
            self.llm_settings = DEFAULT_SETTINGS
            self._settings_warning = "Không thể đọc cấu hình LLM đã lưu; đang dùng Local mặc định."
        self.secret_store = secret_store or SecretStore()
        self.catalog_fetcher = catalog_fetcher

        self.session_memory = self._create_session_memory()
        self.text_bundle: TextRuntimeBundle | None = None
        self.voice_controller: VoiceMonitorController | None = None
        self.voice_stop_event: threading.Event | None = None
        self._voice_threads: list[threading.Thread] = []
        self._chat_thread: threading.Thread | None = None
        self._chat_lock = threading.Lock()
        # Session usage accumulates across chat AND voice turns (shared session).
        self.session_usage = SessionUsage()
        self._usage_lock = threading.Lock()

    # --- lifecycle --------------------------------------------------------------

    def hello(self) -> None:
        stack: dict[str, Any] = {"llm": self.text_config.llm_model}
        if self.voice_config is not None:
            stack = {
                "asr": self.voice_config.asr_model,
                "llm": self.voice_config.llm_model,
                "tts": VALTEC_TTS_CONFIG.key,
                "voice": self.voice_config.tts_voice,
            }
        self.writer.emit(
            {
                "event": "hello",
                "version": PROTOCOL_VERSION,
                "profile": self.profile,
                "no_model": self.no_model,
                "stack": stack,
            }
        )
        if self._settings_warning is not None:
            self._error(self._settings_warning)

    def dispatch(self, command: dict[str, Any]) -> bool:
        """Handle one command; return False when the engine should exit."""
        cmd = command.get("cmd")
        if cmd == "quit":
            return False
        if cmd == "status":
            self._cmd_status()
        elif cmd == "memory":
            self._cmd_memory()
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
        else:
            self._error(f"unknown command: {cmd!r}")
        return True

    def shutdown(self) -> None:
        self._cmd_voice_stop()
        if self._chat_thread is not None:
            self._chat_thread.join(timeout=10.0)
        for thread in self._voice_threads:
            thread.join(timeout=5.0)
        self.writer.emit({"event": "bye"})

    def _error(self, message: str, **extra: Any) -> None:
        self.writer.emit({"event": "engine_error", "message": message, **extra})

    # --- status -----------------------------------------------------------------

    def _cmd_status(self) -> None:
        from soca.app.profiles import collect_runtime_profile_readiness

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
        self.writer.emit({"event": "status", "profiles": profiles})

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
        api_key = self.secret_store.get_key(provider.key)
        if not api_key:
            self._error(f"Chưa có API key cho {provider.label}.")
            return
        try:
            catalog = self.catalog_fetcher(provider, api_key)
        except RemoteLLMError as exc:
            self._error(str(exc), provider=provider.key)
            return
        except Exception as exc:  # noqa: BLE001 - external catalog adapters are untrusted
            LOGGER.warning(
                "Unexpected model catalog failure for provider %s (%s)",
                provider.key,
                type(exc).__name__,
            )
            self._error(
                f"Không thể lấy danh sách model của {provider.label}.", provider=provider.key
            )
            return
        query = command.get("query", "")
        if not isinstance(query, str):
            self._error("LLM model query phải là chuỗi.")
            return
        self.writer.emit(
            {
                "event": "llm_catalog",
                "provider": provider.key,
                "models": [_model_protocol_payload(model) for model in search_models(catalog, query)],
                "pricing_as_of": PRICING_TABLE_AS_OF,
            }
        )

    def _cmd_llm_set_key(self, command: dict[str, Any]) -> None:
        provider = self._provider_from_command(command)
        if provider is None:
            return
        value = command.get("key")
        if not isinstance(value, str) or not value.strip():
            self._emit_key_status(provider.key, ok=False, message="API key không được để trống.")
            return
        secret = value.strip()
        try:
            self.catalog_fetcher(provider, secret)
            self.secret_store.set_key(provider.key, secret)
        except (RemoteLLMError, ValueError) as exc:
            self._emit_key_status(provider.key, ok=False, message=str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - never expose adapter diagnostics or keys to the UI
            LOGGER.warning(
                "Unexpected API-key validation failure for provider %s (%s)",
                provider.key,
                type(exc).__name__,
            )
            self._emit_key_status(
                provider.key,
                ok=False,
                message="Không thể xác thực API key. Hãy thử lại sau.",
            )
            return
        self._emit_key_status(provider.key, ok=True, masked=self.secret_store.mask(secret))

    def _cmd_llm_select(self, command: dict[str, Any]) -> None:
        if self._chat_lock.locked():
            self._error("Không thể đổi LLM khi lượt chat hiện tại đang chạy.")
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
        try:
            settings = LlmSettings(
                backend=backend,
                provider_key=provider_key,
                model_id=model_id,
                max_tokens=self.llm_settings.max_tokens,
                temperature=self.llm_settings.temperature,
                top_p=self.llm_settings.top_p,
            )
        except ValueError as exc:
            self._error(str(exc))
            return
        if settings.backend == "remote" and not self.secret_store.has_key(settings.provider_key):
            provider = get_provider(settings.provider_key)
            self._error(f"Chưa có API key cho {provider.label}.")
            return
        try:
            self.llm_settings_saver(settings)
        except ValueError as exc:
            self._error(str(exc))
            return
        self.llm_settings = settings
        self.text_bundle = None
        self._emit_llm_config()

    def _emit_llm_config(self) -> None:
        settings = self.llm_settings
        active_model = self._active_remote_model(settings)
        payload: dict[str, Any] = {
            "event": "llm_config",
            "backend": settings.backend,
            "provider": settings.provider_key,
            "model": settings.model_id,
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "pricing_as_of": PRICING_TABLE_AS_OF,
            "pricing": active_model,
        }
        self.writer.emit(payload)

    def _active_remote_model(self, settings: LlmSettings) -> RemoteModelInfo | None:
        if settings.backend != "remote":
            return None
        api_key = self.secret_store.get_key(settings.provider_key)
        if not api_key:
            return None
        try:
            provider = get_provider(settings.provider_key)
            catalog = self.catalog_fetcher(provider, api_key)
        except RemoteLLMError:
            return None
        except Exception as exc:  # noqa: BLE001 - config output must not terminate the engine
            LOGGER.warning(
                "Unexpected active-model lookup failure for %s (%s)",
                settings.provider_key,
                type(exc).__name__,
            )
            return None
        return next((model for model in catalog if model.id == settings.model_id), None)

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
    ) -> None:
        payload: dict[str, Any] = {
            "event": "llm_key_status",
            "provider": provider_key,
            "ok": ok,
        }
        if masked is not None:
            payload["masked"] = masked
        if message is not None:
            payload["message"] = message
        self.writer.emit(payload)

    # --- memory & usage -----------------------------------------------------------

    def _cmd_memory(self) -> None:
        if self.session_memory is None:
            self.writer.emit({"event": "memory", "enabled": False, "text": ""})
            return
        rendered = self.session_memory.render().strip()
        self.writer.emit({"event": "memory", "enabled": True, "text": rendered})

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
                    "error_code": None if result.status in {"approved", "rejected"} else result.status,
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
        try:
            self.writer.emit({"event": "chat", "type": "start", "text": text})
            bundle = self._ensure_text_bundle()
            normalized_text, metadata = normalize_text_turn(text)
            result = bundle.runtime.run_text_turn(
                normalized_text, source="engine_chat", metadata=metadata
            )
            usage = TurnUsage.from_runtime_result(result)
            self._track_usage(usage)
            self.writer.emit(
                {
                    "event": "chat",
                    "type": "done",
                    "text": result.response_text,
                    "route": result.route.value,
                    "blocked": result.blocked,
                    "usage": usage,
                }
            )
            trace = result.trace
            if trace is not None:
                commands = self._memory_commands()
                try:
                    pending_count = len(commands.list_pending()) if commands is not None else 0
                except (OSError, ValueError):
                    pending_count = 0
                self.writer.emit(
                    {
                        "event": "router_trace",
                        "tier": trace.tool_router_tier,
                        "tool": trace.tool_calls[-1].name if trace.tool_calls else None,
                        "latency_ms": trace.stage_latencies_ms.get("tool_router", 0.0),
                    }
                )
                knowledge_hits = trace.knowledge_hits[:16]
                if knowledge_hits:
                    self.writer.emit(
                        {
                            "event": "retrieval_trace",
                            "query": result.frame.text if result.frame is not None else "",
                            "tier": trace.tool_router_tier
                            if trace.tool_router_tier in {"deterministic", "semantic", "llm", "none"}
                            else "none",
                            "latency_ms": trace.stage_latencies_ms.get("knowledge", 0.0),
                            "columns": [
                                {
                                    "source": "bm25",
                                    "hits": [
                                        {
                                            "path": str(hit.document.path)[:240],
                                            "score": float(hit.score),
                                        }
                                        for hit in knowledge_hits
                                    ],
                                }
                            ],
                            "fused": [
                                {"path": str(hit.document.path)[:240], "picked": True}
                                for hit in knowledge_hits
                            ],
                        }
                    )
                self.writer.emit(
                    {
                        "event": "memory_trace",
                        "mode": _memory_protocol_mode(
                            trace.memory_mode,
                            trace.memory_degraded_reason,
                            len(trace.memory_hits),
                        ),
                        "degraded_reason": trace.memory_degraded_reason,
                        "hits": [
                            {
                                "id": str(getattr(hit.document, "id", ""))[:120],
                                "corpus": "profile",
                                "relevance": float(getattr(getattr(hit, "score", None), "relevance", 0.0)),
                                "recency": float(getattr(getattr(hit, "score", None), "recency", 0.0)),
                                "importance": float(getattr(getattr(hit, "score", None), "importance", 0.0)),
                                "total": float(getattr(getattr(hit, "score", None), "total", 0.0)),
                            }
                            for hit in trace.memory_hits[:16]
                        ],
                        "compacted_turn_count": 0,
                        "recent_turn_count": len(self.session_memory.turns) if self.session_memory is not None else 0,
                        "background_status": "idle",
                        "episodic_enabled": False,
                        "pending_proposal_count": pending_count,
                    }
                )
        except Exception as exc:  # noqa: BLE001 - protocol boundary must not crash
            self.writer.emit({"event": "chat", "type": "error", "text": str(exc)})
        finally:
            self._chat_lock.release()

    def _ensure_text_bundle(self) -> TextRuntimeBundle:
        if self.text_bundle is not None:
            return self.text_bundle
        self.writer.emit({"event": "chat", "type": "loading", "text": "building text runtime"})
        # Pass the engine's in-memory settings + secret store so chat honours the
        # backend the UI just selected, rather than re-reading disk with a fresh
        # SecretStore. Fall back for builders with a narrower signature (tests).
        try:
            bundle = self.text_runtime_builder(
                self.text_config,
                session_memory=self.session_memory,
                llm_settings=self.llm_settings,
                secret_store=self.secret_store,
            )
        except TypeError:
            try:
                bundle = self.text_runtime_builder(
                    self.text_config, session_memory=self.session_memory
                )
            except TypeError:
                bundle = self.text_runtime_builder(self.text_config)
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

    def _create_session_memory(self) -> SessionMemory | None:
        config = self.text_config
        if config.no_memory:
            return None
        return SessionMemory(
            max_turns=config.session_turns,
            max_chars=config.session_chars,
            max_turn_chars=config.turn_chars,
        )

    # --- voice ------------------------------------------------------------------

    def _cmd_voice_start(self, max_turns: int | None) -> None:
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
            if event.type == "runtime":
                metadata = event.metadata
                tier = metadata.get("router_tier", "none")
                if tier not in {"deterministic", "semantic", "llm", "none"}:
                    tier = "none"
                self.writer.emit(
                    {
                        "event": "router_trace",
                        "tier": tier,
                        "tool": None,
                        "latency_ms": float(metadata.get("router_latency_ms", 0.0)),
                    }
                )
                self.writer.emit(
                    {
                        "event": "memory_trace",
                        "mode": _memory_protocol_mode(
                            metadata.get("memory_mode", "blob"),
                            metadata.get("memory_degraded_reason", ""),
                            metadata.get("memory_hit_count", 0),
                        ),
                        "degraded_reason": metadata.get("memory_degraded_reason", ""),
                        "hits": [],
                        "compacted_turn_count": 0,
                        "recent_turn_count": len(self.session_memory.turns)
                        if self.session_memory is not None
                        else 0,
                        "background_status": "idle",
                        "episodic_enabled": False,
                        "pending_proposal_count": 0,
                    }
                )
            self._track_usage(event.usage)
        if self.voice_stop_event is not None:
            self.voice_stop_event.set()

    def _cmd_voice_stop(self) -> None:
        if self.voice_stop_event is not None:
            self.voice_stop_event.set()
        if self.voice_controller is not None:
            self.voice_controller.stop()

    def _ensure_voice_controller(self) -> VoiceMonitorController:
        if self.voice_controller is not None:
            return self.voice_controller
        assert self.voice_config is not None

        kwargs: dict[str, Any] = {}
        if self.voice_runtime_builder is not None:
            kwargs["runtime_builder"] = self.voice_runtime_builder
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
