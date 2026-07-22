"""Headless SoCa engine: NDJSON protocol over stdio for external UIs.

The Ink TUI (``ui/``) spawns ``soca engine`` and speaks this protocol:

* stdin  — one JSON command per line:
  ``{"cmd": "status"}`` · ``{"cmd": "chat", "text": "..."}`` ·
  ``{"cmd": "voice_start", "max_turns": null}`` · ``{"cmd": "voice_stop"}`` ·
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
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from queue import Queue
from typing import Any, TextIO

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
from soca.core import AudioSink, ResolvedVoiceRuntimeConfig
from soca.core.usage import SessionUsage, TurnUsage
from soca.memory import SessionMemory
from soca.tts import VALTEC_TTS_CONFIG

PROTOCOL_VERSION = 1

TextRuntimeBuilder = Callable[..., TextRuntimeBundle]


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

    def dispatch(self, command: dict[str, Any]) -> bool:
        """Handle one command; return False when the engine should exit."""
        cmd = command.get("cmd")
        if cmd == "quit":
            return False
        if cmd == "status":
            self._cmd_status()
        elif cmd == "memory":
            self._cmd_memory()
        elif cmd == "usage":
            self._cmd_usage()
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

    # --- memory & usage -----------------------------------------------------------

    def _cmd_memory(self) -> None:
        if self.session_memory is None:
            self.writer.emit({"event": "memory", "enabled": False, "text": ""})
            return
        rendered = self.session_memory.render().strip()
        self.writer.emit({"event": "memory", "enabled": True, "text": rendered})

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
        except Exception as exc:  # noqa: BLE001 - protocol boundary must not crash
            self.writer.emit({"event": "chat", "type": "error", "text": str(exc)})
        finally:
            self._chat_lock.release()

    def _ensure_text_bundle(self) -> TextRuntimeBundle:
        if self.text_bundle is not None:
            return self.text_bundle
        self.writer.emit({"event": "chat", "type": "loading", "text": "building text runtime"})
        try:
            bundle = self.text_runtime_builder(self.text_config, session_memory=self.session_memory)
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
