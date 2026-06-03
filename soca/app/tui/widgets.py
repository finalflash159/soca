from __future__ import annotations

from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.selection import Selection
from textual.strip import Strip
from textual.widgets import Input, RichLog, Static

from soca.app.tui.branding import DISPLAY_NAME, compact_header
from soca.app.tui.commands import SlashCommand
from soca.app.tui.events import TuiStageEvent
from soca.app.tui.theme import ACCENT as _ACCENT
from soca.app.tui.theme import BORDER as _BORDER
from soca.app.tui.theme import TITLE as _TITLE
from soca.app.tui.theme import st as _st
from soca.app.usage_view import format_turn_usage_line
from soca.core.turn import RuntimeResult
from soca.core.usage import SessionUsage, TurnUsage


def _inspector_table(title: str) -> Table:
    """Calm-themed table for the Inspector — soft title, muted border, no box noise."""
    return Table(
        title=title,
        title_style=_st(_TITLE),
        border_style=_st(_BORDER) or "none",
        box=box.SIMPLE,
        expand=True,
        pad_edge=False,
    )


class StatusLineWidget(Static):
    def set_status(
        self,
        *,
        mode: str,
        profile: str,
        llm_model: str | None,
        vault_status: str,
        memory_status: str,
        runtime_state: str,
    ) -> None:
        # No forced color here so the muted CSS `color` on #statusline applies
        # (dark bar + soft text, not a filled accent line).
        self.update(
            Text(
                compact_header(
                    mode=mode,
                    profile=profile,
                    llm_model=llm_model,
                    vault_status=vault_status,
                    memory_status=memory_status,
                    runtime_state=runtime_state,
                )
            )
        )


class SidebarWidget(Static):
    """Persistent mode list — a fixed spatial anchor for the cockpit."""

    MODES = (("chat", "Chat"), ("voice", "Voice"), ("status", "Status"))
    active: str = "status"

    def set_mode(self, active: str) -> None:
        self.active = active
        body = Text()
        body.append(f"{DISPLAY_NAME}\n\n", style=_st("bold #3fb950"))
        for key, label in self.MODES:
            is_active = key == active
            body.append(
                f"{'▸ ' if is_active else '  '}{label}\n",
                style=_st("bold #58a6ff") if is_active else _st("#6e7681"),
            )
        self.update(body)


class TimelineWidget(RichLog):
    """Conversation log as clean prefixed lines (no per-message boxes)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._plain_lines: list[str] = []

    def _remember(self, text: str = "") -> None:
        self._plain_lines.append(text)

    @property
    def plain_text(self) -> str:
        return "\n".join(self._plain_lines).strip()

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        return selection.extract(self.plain_text), "\n"

    def _render_line(self, y: int, scroll_x: int, width: int) -> Strip:
        line = super()._render_line(y, scroll_x, width)
        return line.apply_offsets(scroll_x, y)

    def clear(self):
        self._plain_lines.clear()
        return super().clear()

    def _line(self, prefix: str, prefix_style: str, text: str, body_style: str = "#c9d1d9") -> None:
        self._remember(f"{prefix}{text or '<empty>'}")
        line = Text()
        line.append(prefix, style=_st(prefix_style))
        line.append(text or "<empty>", style=_st(body_style))
        self.write(line)

    def add_splash(self, *, bird: str, info: list[str]) -> None:
        grid = Table.grid(padding=(0, 2))
        grid.add_column()
        grid.add_column()
        info_text = Text()
        for index, line in enumerate(info):
            suffix = "\n" if index < len(info) - 1 else ""
            style = _st("bold #c9d1d9") if index == 0 else _st("#6e7681")
            info_text.append(line + suffix, style=style)
        grid.add_row(Text(bird, style=_st("#3fb950")), info_text)
        self.write(grid)
        self.write("")
        self._remember("\n".join([bird, *info]))
        self._remember("")

    def add_system(self, text: str) -> None:
        self._remember(text)
        self.write(Text(text, style=_st("#6e7681")))

    def add_command(self, text: str) -> None:
        self._remember(f"> {text}")
        self.write(Text(f"› {text}", style=_st("#58a6ff")))

    def add_notice(self, title: str, text: str) -> None:
        self._remember(f"{title}: {text}")
        line = Text()
        line.append(f"• {title}: ", style=_st("bold #6e7681"))
        line.append(text, style=_st("#c9d1d9"))
        self.write(line)

    def add_user(self, text: str) -> None:
        self._line("You ▸ ", "bold #58a6ff", text)

    def add_assistant(self, text: str) -> None:
        self._line("(o> SoCa ▸ ", "bold #3fb950", text)

    def add_error(self, text: str) -> None:
        self._line("✗ ", "bold #f85149", text, body_style="#f85149")


class StageRailWidget(Static):
    def show_idle(self) -> None:
        self.update(Text("stage: idle", style=_st("#6e7681")))

    def show_stages(self, events: tuple[TuiStageEvent, ...]) -> None:
        if not events:
            self.show_idle()
            return
        parts = []
        for event in events:
            suffix = f" {event.latency_ms:.0f}ms" if event.latency_ms is not None else ""
            detail = f" ({event.detail})" if event.detail else ""
            parts.append(f"{event.stage}:{event.status}{suffix}{detail}")
        self.update(Text(" -> ".join(parts), style=_st(_ACCENT)))


class InspectorWidget(Static):
    def show_idle(self, text: str = "Chưa có turn nào.") -> None:
        self.update(
            Panel(
                Text(text, style=_st("#c9d1d9")),
                title=Text("Inspector", style=_st(_TITLE)),
                border_style=_st(_BORDER) or "none",
                padding=(0, 1),
            )
        )

    def show_status_summary(self, rows: list[tuple[str, str, str]]) -> None:
        table = _inspector_table("Runtime readiness")
        table.add_column("Profile", style=_st(_ACCENT), no_wrap=True)
        table.add_column("Status", no_wrap=True)
        table.add_column("Stack", overflow="fold")
        for profile, status, stack in rows:
            table.add_row(profile, status, stack)
        self.update(table)

    def show_result(self, result: RuntimeResult, usage: TurnUsage | None = None) -> None:
        table = _inspector_table("Turn Inspector")
        table.add_column("Field", style=_st(_ACCENT), no_wrap=True)
        table.add_column("Value", overflow="fold")

        trace = result.trace
        table.add_row("route", result.route.value)
        table.add_row("blocked", str(result.blocked))
        if trace is not None:
            table.add_row("used_tool", str(trace.used_tool))
            table.add_row("used_llm", str(trace.used_llm))
            if trace.tool_calls:
                table.add_row("tools", ", ".join(call.name for call in trace.tool_calls))
            if trace.guardrail_events:
                table.add_row(
                    "guardrails",
                    ", ".join(
                        f"{event.stage.value}:{event.action.value}"
                        for event in trace.guardrail_events
                    ),
                )
            if trace.stage_latencies_ms:
                table.add_row(
                    "latency",
                    ", ".join(
                        f"{stage}={latency_ms:.0f}ms"
                        for stage, latency_ms in trace.stage_latencies_ms.items()
                    ),
                )
        if result.citations:
            table.add_row(
                "citations",
                ", ".join(
                    f"[K{index}] {citation.path}"
                    for index, citation in enumerate(result.citations, start=1)
                ),
            )
        if usage is not None:
            table.add_row("usage", format_turn_usage_line(usage))

        self.update(table)

    def show_voice_summary(
        self,
        *,
        transcript: str,
        response: str,
        route: str,
        rejected: bool,
        blocked: bool,
        stage_latencies_ms: dict[str, float] | None = None,
        usage: TurnUsage | None = None,
    ) -> None:
        table = _inspector_table("Voice Turn")
        table.add_column("Field", style=_st(_ACCENT), no_wrap=True)
        table.add_column("Value", overflow="fold")
        table.add_row("transcript", transcript or "<empty>")
        table.add_row("route", route or "unknown")
        table.add_row("rejected", str(rejected))
        table.add_row("blocked", str(blocked))
        if response:
            table.add_row("response", response)
        if stage_latencies_ms:
            table.add_row(
                "latency",
                ", ".join(f"{stage}={latency_ms:.0f}ms" for stage, latency_ms in stage_latencies_ms.items()),
            )
        if usage is not None:
            table.add_row("usage", format_turn_usage_line(usage))
            if usage.tts_chunks is not None:
                table.add_row("tts_chunks", str(usage.tts_chunks))
            if usage.ttfa_ms is not None:
                table.add_row("ttfa", f"{usage.ttfa_ms:.0f}ms")
        self.update(table)

    def show_usage(self, session: SessionUsage) -> None:
        table = _inspector_table("Session Usage")
        table.add_column("Metric", style=_st(_ACCENT))
        table.add_column("Value", justify="right")
        table.add_row("turns", str(session.total_turns))
        table.add_row("LLM turns", str(session.llm_turns))
        table.add_row("prompt tokens", str(session.total_prompt_tokens))
        table.add_row("completion tokens", str(session.total_completion_tokens))
        table.add_row("mean TTFT", f"{session.mean_ttft_ms:.0f} ms")
        table.add_row("mean tok/s", f"{session.mean_tokens_per_second:.1f}")
        self.update(table)


class ComposerWidget(Input):
    pass


class SlashCommandListWidget(Static):
    last_text: str = ""

    def show_matches(self, query: str, commands: tuple[SlashCommand, ...]) -> None:
        if not query.startswith("/"):
            self.display = False
            self.last_text = ""
            return

        self.display = True
        table = Table.grid(padding=(0, 2))
        table.add_column("Command", style=_st(f"bold {_ACCENT}"), no_wrap=True)
        table.add_column("Description", style=_st("#c9d1d9"))

        if commands:
            for command in commands:
                table.add_row(command.name, command.description)
        else:
            table.add_row("no match", "gõ /help để xem toàn bộ lệnh")

        self.last_text = "\n".join(
            f"{command.name} {command.description}" for command in commands
        )
        title = Text()
        title.append("Slash ", style=_st(_TITLE))
        title.append(query, style=_st("#6e7681"))
        self.update(
            Panel(
                table,
                title=title,
                border_style=_st(_BORDER) or "none",
                padding=(0, 1),
            )
        )

    def hide(self) -> None:
        self.display = False
        self.last_text = ""


__all__ = [
    "ComposerWidget",
    "InspectorWidget",
    "SidebarWidget",
    "SlashCommandListWidget",
    "StageRailWidget",
    "StatusLineWidget",
    "TimelineWidget",
]
