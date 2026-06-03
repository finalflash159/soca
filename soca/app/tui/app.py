from __future__ import annotations

import asyncio
import inspect
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Queue
from threading import Event as ThreadEvent
from threading import Thread

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Input

from soca.app.profiles import collect_runtime_profile_readiness
from soca.app.text_chat import _render_session_memory
from soca.app.text_runtime import (
    TextRuntimeBundle,
    TextRuntimeConfig,
    build_text_runtime,
    normalize_text_turn,
)
from soca.app.tui.branding import BIRD_MARK
from soca.app.tui.commands import filter_slash_commands, slash_help_text
from soca.app.tui.events import TuiStageEvent
from soca.app.tui.state import TuiMode, TuiState
from soca.app.tui.voice import (
    VoiceMonitorController,
    VoiceMonitorEvent,
    VoiceRecorder,
    VoiceRuntimeBuilder,
)
from soca.app.tui.voice_view import VoiceStatusBar, VoiceTurnView
from soca.app.tui.widgets import (
    ComposerWidget,
    InspectorWidget,
    SidebarWidget,
    SlashCommandListWidget,
    StageRailWidget,
    StatusLineWidget,
    TimelineWidget,
)
from soca.core import DEFAULT_VOICE_RUNTIME_PROFILE_KEY, AudioSink, ResolvedVoiceRuntimeConfig
from soca.core.usage import SessionUsage, TurnUsage
from soca.memory import SessionMemory

CHAT_EXIT_COMMANDS = {"/exit", "/quit", ":q"}
TUI_HELP = slash_help_text()

RuntimeBuilder = Callable[..., TextRuntimeBundle]


@dataclass(frozen=True)
class TuiConfig:
    mode: TuiMode = "status"
    profile: str = DEFAULT_VOICE_RUNTIME_PROFILE_KEY
    text_runtime: TextRuntimeConfig = TextRuntimeConfig()
    voice_runtime: ResolvedVoiceRuntimeConfig | None = None
    no_model: bool = False
    show_splash: bool = True
    warmup_voice: bool = True
    auto_start_voice: bool = True
    voice_loop_max_turns: int | None = None


class SoCaTuiApp(App[None]):
    """Optional Textual cockpit for SoCa.

    The TUI is intentionally a client. It renders runtime state and calls the
    existing text runtime builder; it does not own routing, search, prompt
    construction, model registries, or guardrail policy.
    """

    CSS_PATH = "styles.tcss"

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("t", "toggle_trace", "Trace"),
        ("u", "show_usage", "Usage"),
        ("y", "copy_timeline", "Copy"),
        ("c", "clear_timeline", "Clear"),
        ("?", "help", "Help"),
    ]

    def __init__(
        self,
        config: TuiConfig | None = None,
        *,
        runtime_builder: RuntimeBuilder = build_text_runtime,
        voice_runtime_builder: VoiceRuntimeBuilder | None = None,
        voice_recorder: VoiceRecorder | None = None,
        voice_player: AudioSink | None = None,
    ) -> None:
        super().__init__()
        self.config = config or TuiConfig()
        self.runtime_builder = runtime_builder
        self.voice_runtime_builder = voice_runtime_builder
        self.voice_recorder = voice_recorder
        self.voice_player = voice_player
        self.shared_session_memory = self._create_shared_session_memory()
        self.state = TuiState(mode=self.config.mode, profile=self.config.profile)
        self.bundle: TextRuntimeBundle | None = None
        self.voice_controller: VoiceMonitorController | None = None
        self.voice_running = False
        self.voice_stop_event: ThreadEvent | None = None
        self.voice_consumer_task: asyncio.Task[None] | None = None
        self.last_result = None
        self._chat_busy = False
        self._voice_transcript = ""
        self._voice_response = ""
        self._voice_runtime_meta: dict[str, object] = {}
        self._voice_turn = VoiceTurnView()
        self._music_frame = 0
        self._pending_handover: str | None = None

    def compose(self) -> ComposeResult:
        yield StatusLineWidget(id="statusline")
        with Horizontal(id="main"):
            yield SidebarWidget(id="sidebar")
            with Vertical(id="content"):
                yield StageRailWidget(id="stage_rail")
                yield VoiceStatusBar(id="voice_status")
                with Horizontal(id="body"):
                    yield TimelineWidget(id="timeline", wrap=True, highlight=False, markup=False)
                    yield InspectorWidget(id="inspector")
                yield SlashCommandListWidget(id="slash_commands")
                yield ComposerWidget(
                    placeholder=self._composer_placeholder(self.config.mode),
                    id="composer",
                )
        yield Footer()

    def on_mount(self) -> None:
        self.sidebar.set_mode(self.config.mode)
        self._apply_layout(self.config.mode)
        self.set_interval(0.4, self._tick_voice_music)
        self._refresh_status("idle")
        if self.config.show_splash:
            self.timeline.add_splash(bird=BIRD_MARK, info=self._splash_info())

        if self.config.mode == "status":
            self._render_status_mode()
        elif self.config.mode == "voice":
            self._render_voice_placeholder()
            if self.config.auto_start_voice and not self.config.no_model:
                asyncio.get_running_loop().create_task(self._start_voice_loop())
        else:
            self.timeline.add_notice("Chat", "Sẵn sàng. Gõ /help để xem lệnh.")
            self.inspector.show_idle("Runtime sẽ được build ở turn đầu tiên.")
            self.stage_rail.show_idle()
        self.slash_commands.hide()
        self.query_one("#composer", ComposerWidget).focus()

    @property
    def timeline(self) -> TimelineWidget:
        return self.query_one("#timeline", TimelineWidget)

    @property
    def inspector(self) -> InspectorWidget:
        return self.query_one("#inspector", InspectorWidget)

    @property
    def statusline(self) -> StatusLineWidget:
        return self.query_one("#statusline", StatusLineWidget)

    @property
    def stage_rail(self) -> StageRailWidget:
        return self.query_one("#stage_rail", StageRailWidget)

    @property
    def slash_commands(self) -> SlashCommandListWidget:
        return self.query_one("#slash_commands", SlashCommandListWidget)

    @property
    def sidebar(self) -> SidebarWidget:
        return self.query_one("#sidebar", SidebarWidget)

    @property
    def voice_status(self) -> VoiceStatusBar:
        return self.query_one("#voice_status", VoiceStatusBar)

    def _apply_layout(self, mode: TuiMode) -> None:
        """Voice keeps the same timeline|inspector body as chat; it only adds a
        thin live status strip on top. The plain stage rail is for chat/status."""
        is_voice = mode == "voice"
        self.voice_status.display = is_voice
        self.stage_rail.display = not is_voice

    def _set_voice_turn(self, **changes: object) -> None:
        """Apply an immutable update to the voice snapshot and re-render."""
        self._voice_turn = replace(self._voice_turn, **changes)
        self._render_voice_view()

    def _render_voice_view(self) -> None:
        self.voice_status.render_status(
            self._voice_turn,
            profile=self.config.profile,
            memory_on=not self.config.text_runtime.no_memory,
            music_frame=self._music_frame,
        )

    def _tick_voice_music(self) -> None:
        """Animate the music notes while SoCa is speaking (no-op otherwise)."""
        if self.state.mode == "voice" and self._voice_turn.state == "speaking":
            self._music_frame += 1
            self._render_voice_view()

    def action_toggle_trace(self) -> None:
        self.state.trace_enabled = not self.state.trace_enabled
        self.timeline.add_notice("Trace", "on" if self.state.trace_enabled else "off")

    def action_show_usage(self) -> None:
        self.inspector.show_usage(self.state.session_usage)

    def action_clear_timeline(self) -> None:
        self.timeline.clear()
        self.inspector.show_idle("Timeline đã được xóa.")

    def action_help(self) -> None:
        self.timeline.add_notice("Help", TUI_HELP)

    def action_copy_timeline(self) -> None:
        text = self.timeline.plain_text
        if not text:
            self.timeline.add_notice("Copy", "Timeline đang trống.")
            return
        self.copy_to_clipboard(text)
        self.timeline.add_notice("Copy", "Đã copy transcript.")

    def copy_to_clipboard(self, text: str) -> None:
        """Copy through Textual and the native macOS clipboard when available.

        Textual writes OSC52 plus an internal clipboard. OSC52 is terminal-dependent
        and Textual explicitly notes that macOS Terminal does not support it, so
        SoCa adds a small pbcopy fallback for the common local macOS path.
        """
        super().copy_to_clipboard(text)
        if sys.platform != "darwin":
            return
        try:
            subprocess.run(
                ["pbcopy"],
                input=text,
                text=True,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            return

    async def action_listen(self) -> None:
        await self._start_voice_loop()

    def on_unmount(self) -> None:
        self._request_voice_stop(quiet=True)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        event.input.value = ""
        self.slash_commands.hide()
        if not user_text:
            return
        await self._handle_chat_input(user_text)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "composer":
            return
        value = event.value.strip()
        self.slash_commands.show_matches(value, filter_slash_commands(value))

    async def _handle_chat_input(self, user_text: str) -> None:
        command = user_text.lower()
        is_command = user_text.startswith("/") or command in CHAT_EXIT_COMMANDS
        if is_command:
            self.timeline.add_command(user_text)
        if command in CHAT_EXIT_COMMANDS:
            self.exit()
            return
        if command == "/help":
            self.timeline.add_notice("Help", TUI_HELP)
            return
        if command == "/status":
            self._request_voice_stop()
            self._switch_mode("status")
            return
        if command == "/chat":
            self._request_voice_stop()
            self._switch_mode("chat")
            return
        if command == "/voice":
            self._switch_mode("voice")
            if self.config.auto_start_voice:
                await self._start_voice_loop()
            return
        if command == "/listen":
            await self._start_voice_loop()
            return
        if command == "/stop":
            self._request_voice_stop()
            return
        if command == "/trace":
            self.action_toggle_trace()
            return
        if command == "/usage":
            self.inspector.show_usage(self.state.session_usage)
            return
        if command == "/copy":
            self.action_copy_timeline()
            return
        if command == "/clear":
            if self.shared_session_memory is not None:
                self.shared_session_memory.clear()
            elif self.bundle and self.bundle.session_memory:
                self.bundle.session_memory.clear()
            self.state.session_usage = SessionUsage()
            self.timeline.clear()
            self.inspector.show_idle("Session đã được xóa.")
            return
        if command == "/memory":
            if self.bundle is None:
                self.inspector.show_idle("Runtime chưa được build nên memory chưa có dữ liệu.")
            else:
                self.inspector.show_idle(_render_session_memory(self.bundle))
            return

        if self.state.mode != "chat":
            self.timeline.add_notice(
                "Read-only mode",
                f"Bạn đang ở mode `{self.state.mode}`. Gõ /chat để gửi câu hỏi vào runtime.",
            )
            return

        # llama.cpp/AssistantRuntime is not safe to call concurrently; block a
        # second submit (and disable input) until the current turn finishes.
        if self._chat_busy:
            self.timeline.add_notice("Chat", "Đang xử lý câu trước, đợi một chút...")
            return

        self.timeline.add_user(user_text)
        self._refresh_status("processing")
        self.stage_rail.show_stages((TuiStageEvent("input", "ok"), TuiStageEvent("runtime", "running")))

        composer = self.query_one("#composer", ComposerWidget)
        self._chat_busy = True
        composer.disabled = True
        try:
            bundle = await self._ensure_bundle()
            normalized_text, metadata = normalize_text_turn(user_text)
            result = await asyncio.to_thread(
                bundle.runtime.run_text_turn,
                normalized_text,
                source="tui_chat",
                metadata=metadata,
            )
        except Exception as exc:  # pragma: no cover - terminal boundary
            self._refresh_status("error")
            self.timeline.add_error(str(exc))
            self.inspector.show_idle(str(exc))
            return
        finally:
            self._chat_busy = False
            composer.disabled = False
            composer.focus()

        self.last_result = result
        self.timeline.add_assistant(result.response_text)
        turn_usage = TurnUsage.from_runtime_result(result)
        self.state.session_usage = self.state.session_usage.add(turn_usage)
        self.state.last_turn_usage = turn_usage
        self.inspector.show_result(result, usage=turn_usage)
        latency = turn_usage.runtime_latency_ms
        self.stage_rail.show_stages(
            (
                TuiStageEvent("input", "ok"),
                TuiStageEvent("runtime", "ok", latency_ms=latency),
                TuiStageEvent("output", "ok"),
            )
        )
        self._refresh_status("idle")

    async def _ensure_bundle(self) -> TextRuntimeBundle:
        if self.bundle is not None:
            return self.bundle

        self._refresh_status("loading-runtime")
        config = self.config.text_runtime
        if self.config.no_model and not config.no_llm:
            config = replace(config, no_llm=True)
        self.bundle = await asyncio.to_thread(self._build_text_runtime_bundle, config)
        # Runtime readiness is persistent state, not a conversation event — it
        # belongs in the statusline/Inspector, not inline in the chat timeline.
        self.inspector.show_idle(
            "Runtime ready\n"
            f"LLM={self.bundle.llm_status}\n"
            f"Knowledge={self.bundle.knowledge_status}\n"
            f"Memory={self.bundle.memory_status}"
        )
        return self.bundle

    def _create_shared_session_memory(self) -> SessionMemory | None:
        config = self.config.text_runtime
        if config.no_memory:
            return None
        return SessionMemory(
            max_turns=config.session_turns,
            max_chars=config.session_chars,
            max_turn_chars=config.turn_chars,
        )

    def _build_text_runtime_bundle(self, config: TextRuntimeConfig) -> TextRuntimeBundle:
        if self.shared_session_memory is None:
            return self.runtime_builder(config)

        try:
            signature = inspect.signature(self.runtime_builder)
        except (TypeError, ValueError):
            signature = None

        supports_session_memory = (
            signature is not None
            and (
                "session_memory" in signature.parameters
                or any(
                    param.kind is inspect.Parameter.VAR_KEYWORD
                    for param in signature.parameters.values()
                )
            )
        )
        if supports_session_memory:
            return self.runtime_builder(config, session_memory=self.shared_session_memory)
        return self.runtime_builder(config)

    async def _start_voice_loop(self) -> None:
        if self.config.no_model:
            if self.state.mode != "voice":
                self._switch_mode("voice")
            self._set_voice_turn(
                state="idle",
                note="`--no-model` đang bật nên TUI không load ASR/LLM/TTS. Tắt flag này để chạy voice loop.",
            )
            return
        if self.config.voice_runtime is None:
            if self.state.mode != "voice":
                self._switch_mode("voice")
            self._set_voice_turn(state="error", note="Voice runtime config chưa được truyền vào TUI.")
            return
        if self.voice_running:
            self._set_voice_turn(note="Voice loop đang chạy.")
            return

        if self.state.mode != "voice":
            self._switch_mode("voice")
        self.voice_running = True
        self.voice_stop_event = ThreadEvent()
        self._voice_transcript = ""
        self._voice_response = ""
        self._voice_runtime_meta = {}
        self._voice_turn = VoiceTurnView(state="listening", note="Đang khởi động voice loop…")
        self._render_voice_view()
        self._refresh_status("voice")

        event_queue: Queue[VoiceMonitorEvent | None] = Queue()
        worker = Thread(
            target=self._run_voice_loop_worker,
            args=(event_queue,),
            daemon=True,
            name="soca-tui-voice-loop",
        )
        worker.start()
        self.voice_consumer_task = asyncio.create_task(self._consume_voice_events(event_queue))

    async def _consume_voice_events(self, event_queue: Queue[VoiceMonitorEvent | None]) -> None:
        while True:
            item = await asyncio.to_thread(event_queue.get)
            if item is None:
                break
            self._render_voice_event(item)

        self.voice_running = False
        self.voice_stop_event = None
        self.voice_consumer_task = None
        self._refresh_status("idle")

    def _run_voice_loop_worker(self, event_queue: Queue[VoiceMonitorEvent | None]) -> None:
        controller = self._ensure_voice_controller()
        stop_event = self.voice_stop_event or ThreadEvent()
        controller.run_loop(
            event_queue,
            stop_event=stop_event,
            max_turns=self.config.voice_loop_max_turns,
        )

    def _request_voice_stop(self, *, quiet: bool = False) -> None:
        if not self.voice_running:
            return
        if self.voice_stop_event is not None:
            self.voice_stop_event.set()
        if self.voice_controller is not None:
            self.voice_controller.stop()
        if not quiet:
            self._set_voice_turn(note="Đã nhận /stop. Voice loop sẽ dừng sau turn hiện tại.")

    def _ensure_voice_controller(self) -> VoiceMonitorController:
        if self.voice_controller is not None:
            return self.voice_controller
        if self.config.voice_runtime is None:
            raise RuntimeError("Voice runtime config is missing")

        kwargs = {}
        if self.voice_runtime_builder is not None:
            kwargs["runtime_builder"] = self.voice_runtime_builder
        if self.voice_recorder is not None:
            kwargs["recorder"] = self.voice_recorder
        if self.voice_player is not None:
            kwargs["player"] = self.voice_player

        self.voice_controller = VoiceMonitorController(
            self.config.voice_runtime,
            warmup=self.config.warmup_voice,
            session_memory=self.shared_session_memory,
            **kwargs,
        )
        return self.voice_controller

    # Voice event type -> renderer method name. Replaces a long if-elif chain
    # with an explicit dispatch table.
    _VOICE_RENDERERS = {
        "loading": "_voice_on_loading",
        "ready": "_voice_on_ready",
        "warmup": "_voice_on_warmup",
        "loop_started": "_voice_on_loop_started",
        "turn_start": "_voice_on_turn_start",
        "recording": "_voice_on_recording",
        "recorded": "_voice_on_recorded",
        "idle_silence": "_voice_on_idle_silence",
        "idle_sleep": "_voice_on_idle_sleep",
        "asr": "_voice_on_asr",
        "repair": "_voice_on_repair",
        "llm_token": "_voice_on_llm_token",
        "runtime": "_voice_on_runtime",
        "sentence": "_voice_on_sentence",
        "tts": "_voice_on_tts",
        "audio": "_voice_on_audio",
        "done": "_render_voice_done",
        "turn_end": "_voice_on_turn_end",
        "loop_stopped": "_voice_on_loop_stopped",
        "error": "_voice_on_error",
    }

    def _render_voice_event(self, event: VoiceMonitorEvent) -> None:
        handler_name = self._VOICE_RENDERERS.get(event.type)
        if handler_name is None:
            self._set_voice_turn(note=f"{event.type}: {event.text}")
            return
        getattr(self, handler_name)(event)

    # Operational events go to the Inspector / live status strip, never the
    # timeline. Only the actual conversation (ASR line, SoCa reply) hits the
    # timeline so voice history reads like chat and survives a mode switch.

    def _voice_on_loading(self, event: VoiceMonitorEvent) -> None:
        self.inspector.show_idle(f"Đang load ASR/LLM/TTS.\n{_voice_stack_text(event.metadata)}")
        self._set_voice_turn(state="idle", note="đang tải…")

    def _voice_on_ready(self, event: VoiceMonitorEvent) -> None:
        self.inspector.show_idle(
            "Voice runtime sẵn sàng.\n"
            f"Memory={event.metadata.get('memory_status', 'unknown')}\n"
            f"Knowledge={event.metadata.get('knowledge_status', 'unknown')}"
        )
        self._set_voice_turn(note="")

    def _voice_on_warmup(self, event: VoiceMonitorEvent) -> None:
        status = "ok" if event.metadata.get("ok") else "error"
        self._set_voice_turn(note=f"warmup {event.text}: {status}")

    def _voice_on_loop_started(self, event: VoiceMonitorEvent) -> None:
        self._set_voice_turn(state="listening", note="")
        self._refresh_status("listening")

    def _voice_on_turn_start(self, event: VoiceMonitorEvent) -> None:
        turn_index = event.metadata.get("turn_index")
        self._voice_transcript = ""
        self._voice_response = ""
        self._voice_runtime_meta = {}
        self._voice_turn = VoiceTurnView(
            state="listening",
            turn_index=int(turn_index) if isinstance(turn_index, int) else None,
        )
        self._render_voice_view()

    def _voice_on_recording(self, event: VoiceMonitorEvent) -> None:
        self._set_voice_turn(state="listening", note="đang nghe…")
        self._refresh_status("listening")

    def _voice_on_recorded(self, event: VoiceMonitorEvent) -> None:
        self._set_voice_turn(state="processing", note="")
        self._refresh_status("processing")

    def _voice_on_idle_silence(self, event: VoiceMonitorEvent) -> None:
        self._set_voice_turn(state="idle", note="im lặng, tiếp tục nghe…")
        self._refresh_status("listening")

    def _voice_on_idle_sleep(self, event: VoiceMonitorEvent) -> None:
        self._set_voice_turn(state="idle", note="ngủ sau một lúc im lặng")
        self.inspector.show_idle("Voice tạm dừng vì không có hội thoại mới.")
        self._refresh_status("idle")

    def _voice_on_asr(self, event: VoiceMonitorEvent) -> None:
        self._voice_transcript = event.text
        # The ASR transcript IS the user's turn — show it in the timeline like chat.
        self.timeline.add_user(event.text or "<empty>")
        rejection = event.metadata.get("rejection_reason")
        self.inspector.show_idle(
            f"ASR: {event.text or '<empty>'}\nrejection={rejection or 'none'}"
        )
        self._set_voice_turn(state="processing")

    def _voice_on_repair(self, event: VoiceMonitorEvent) -> None:
        # A conversation-repair follow-up (not an error): show it as SoCa's turn
        # and remember any handover so we can switch channels once it's spoken.
        self.timeline.add_notice("Follow-up", event.text or "")
        self._voice_response = event.text
        handover = event.metadata.get("handover_target")
        self._pending_handover = str(handover) if handover else None
        self.inspector.show_idle(
            f"Repair: {event.metadata.get('repair_kind', '?')}"
            f" · {event.metadata.get('repair_action', '?')}"
            f" · attempt {event.metadata.get('repair_attempt', '?')}\n"
            f"reason={event.metadata.get('technical_reason', '?')}"
        )
        self._set_voice_turn(state="processing", note="follow-up")

    def _voice_on_llm_token(self, event: VoiceMonitorEvent) -> None:
        self._voice_response += event.text
        self.inspector.show_idle(f"SoCa đang soạn:\n{self._voice_response[-400:]}")
        self._set_voice_turn(state="processing")

    def _voice_on_runtime(self, event: VoiceMonitorEvent) -> None:
        self._voice_response = event.text or self._voice_response
        self._voice_runtime_meta = event.metadata

    def _voice_on_sentence(self, event: VoiceMonitorEvent) -> None:
        return

    def _voice_on_tts(self, event: VoiceMonitorEvent) -> None:
        self._set_voice_turn(state="speaking")
        self._refresh_status("speaking")

    def _voice_on_audio(self, event: VoiceMonitorEvent) -> None:
        return

    def _voice_on_turn_end(self, event: VoiceMonitorEvent) -> None:
        self._set_voice_turn(state="idle")

    def _voice_on_loop_stopped(self, event: VoiceMonitorEvent) -> None:
        self._set_voice_turn(
            state="idle",
            note=f"đã dừng ({event.metadata.get('turns', 0)} lượt)",
        )
        self._refresh_status("idle")

    def _voice_on_error(self, event: VoiceMonitorEvent) -> None:
        self.timeline.add_error(event.text)
        details = event.text
        traceback_text = event.metadata.get("traceback")
        if traceback_text:
            details = f"{details}\n\nTraceback:\n{traceback_text}"
        self.inspector.show_idle(details)
        self._set_voice_turn(state="error", note="lỗi")
        self._refresh_status("error")

    def _render_voice_done(self, event: VoiceMonitorEvent) -> None:
        metadata = event.metadata
        response = event.text or self._voice_response
        rejected = bool(metadata.get("rejected"))
        route = str(metadata.get("runtime_route") or self._voice_runtime_meta.get("route") or "unknown")
        blocked = bool(metadata.get("runtime_blocked") or self._voice_runtime_meta.get("blocked"))
        stage_latencies = metadata.get("stage_latencies_ms")

        if event.usage is not None:
            self.state.session_usage = self.state.session_usage.add(event.usage)
            self.state.last_turn_usage = event.usage

        # SoCa's reply joins the conversation timeline. For a rejected turn the
        # `repair` event already added the Follow-up line, so don't duplicate it.
        if response and not rejected:
            self.timeline.add_assistant(response)

        self.inspector.show_voice_summary(
            transcript=self._voice_transcript,
            response=response,
            route=route,
            rejected=rejected,
            blocked=blocked,
            stage_latencies_ms=stage_latencies if isinstance(stage_latencies, dict) else None,
            usage=event.usage,
        )
        self._set_voice_turn(state="idle", note="")

        # Act on a planned handover once the follow-up has been spoken (plan §R4).
        if self._pending_handover == "chat":
            self._pending_handover = None
            self.timeline.add_notice("Handover", "Voice trục trặc — chuyển sang chat để bạn gõ.")
            self._switch_mode("chat")

    def _render_status_mode(self) -> None:
        rows = []
        for item in collect_runtime_profile_readiness():
            rows.append(
                (
                    item.key,
                    item.profile_status,
                    f"ASR={item.asr_model} | LLM={item.llm_model} | "
                    f"TTS={item.tts_model}/{item.tts_voice or 'default'}",
                )
            )
        self.inspector.show_status_summary(rows)
        self.stage_rail.show_idle()
        self.timeline.add_notice(
            "Status",
            "Không load ASR/LLM/TTS. Gõ /chat để hỏi, /voice để xem voice shell, /help để xem lệnh.",
        )

    def _render_voice_placeholder(self) -> None:
        if self.config.no_model:
            state, note = "idle", "no-model"
            detail = "no-model: TUI không load ASR/LLM/TTS. Bỏ --no-model để record."
        elif self.config.voice_runtime is None:
            state, note = "error", "config thiếu"
            detail = "Voice runtime config chưa sẵn sàng."
        else:
            rt = self.config.voice_runtime
            voice = rt.tts_voice or "default"
            state, note = "idle", ""
            detail = (
                "Voice loop đã sẵn sàng.\n"
                f"ASR={rt.asr_model}\nLLM={rt.llm_model}\nTTS={rt.tts_model}/{voice}\n\n"
                "Loop sẽ tự chạy. /stop để dừng, /listen để chạy lại."
            )
        self.inspector.show_idle(detail)
        self._voice_turn = VoiceTurnView(state=state, note=note)
        self._render_voice_view()

    def _refresh_status(self, runtime_state: str) -> None:
        self.state.runtime_state = runtime_state
        vault = self.config.text_runtime.vault.expanduser()
        vault_status = "ok" if vault.is_dir() else "missing"
        memory_status = "off" if self.config.text_runtime.no_memory else "on"
        if self.config.no_model:
            runtime_state = f"{runtime_state}:no-model"
        self.statusline.set_status(
            mode=self.state.mode,
            profile=self.config.profile,
            llm_model=None if self.config.no_model else self.config.text_runtime.llm_model,
            vault_status=vault_status,
            memory_status=memory_status,
            runtime_state=runtime_state,
        )

    def _switch_mode(self, mode: TuiMode) -> None:
        # Leaving voice must stop the loop so the mic/recorder doesn't keep
        # running behind the chat view. Timeline + session memory are untouched
        # so the conversation history carries across the switch.
        if mode != "voice" and self.voice_running:
            self._request_voice_stop(quiet=True)

        self.state.mode = mode
        self.sidebar.set_mode(mode)
        self._apply_layout(mode)
        self._refresh_status("idle")
        self.query_one("#composer", ComposerWidget).placeholder = self._composer_placeholder(mode)

        if mode == "status":
            self._render_status_mode()
            return
        if mode == "voice":
            self._render_voice_placeholder()
            return

        self.stage_rail.show_idle()
        self.inspector.show_idle("Chat mode. Runtime sẽ được build ở turn đầu tiên.")

    def _splash_info(self) -> list[str]:
        cfg = self.config.text_runtime
        model = "no-model" if self.config.no_model else cfg.llm_model
        vault = str(cfg.vault)
        home = str(Path.home())
        if vault.startswith(home):
            vault = "~" + vault[len(home):]
        return [
            "SoCa · local Vietnamese voice assistant",
            f"profile {self.config.profile} · {model}",
            vault,
        ]

    def _composer_placeholder(self, mode: TuiMode) -> str:
        if mode == "chat":
            return "Chat với SoCa hoặc nhập /help..."
        if mode == "voice":
            return "Voice loop: /stop, /listen, /chat, /status, /usage, /help, /exit..."
        return "Status mode: /chat, /voice, /help, /exit..."


def _voice_stack_text(metadata: dict[str, object]) -> str:
    voice = metadata.get("voice") or "default"
    return (
        f"profile={metadata.get('profile', 'unknown')}\n"
        f"ASR={metadata.get('asr_model', 'unknown')}\n"
        f"LLM={metadata.get('llm_model', 'unknown')}\n"
        f"TTS={metadata.get('tts_model', 'unknown')}/{voice}"
    )


def run_tui(
    *,
    mode: TuiMode = "status",
    profile: str = DEFAULT_VOICE_RUNTIME_PROFILE_KEY,
    text_runtime: TextRuntimeConfig | None = None,
    voice_runtime: ResolvedVoiceRuntimeConfig | None = None,
    no_model: bool = False,
    warmup_voice: bool = True,
    runtime_builder: RuntimeBuilder = build_text_runtime,
    voice_runtime_builder: VoiceRuntimeBuilder | None = None,
    voice_recorder: VoiceRecorder | None = None,
    voice_player: AudioSink | None = None,
) -> int:
    config = TuiConfig(
        mode=mode,
        profile=profile,
        text_runtime=text_runtime or TextRuntimeConfig(),
        voice_runtime=voice_runtime,
        no_model=no_model,
        warmup_voice=warmup_voice,
    )
    SoCaTuiApp(
        config,
        runtime_builder=runtime_builder,
        voice_runtime_builder=voice_runtime_builder,
        voice_recorder=voice_recorder,
        voice_player=voice_player,
    ).run()
    return 0


__all__ = ["SoCaTuiApp", "TuiConfig", "run_tui"]
