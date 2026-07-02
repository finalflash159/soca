from __future__ import annotations

from collections.abc import Callable

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from soca.app.style.palette import ACCENT, ALT, BORDER, ICON, MUTED, TITLE, st
from soca.app.text_runtime import (
    TextRuntimeBundle,
    TextRuntimeConfig,
    build_text_runtime,
    normalize_text_turn,
    render_text_result,
)
from soca.app.usage_view import print_turn_usage, render_session_usage
from soca.core.usage import SessionUsage, TurnUsage

CHAT_EXIT_COMMANDS = {"/exit", "/quit", ":q"}
CHAT_HELP = "\n".join(
    [
        "Nhập câu bình thường để chat qua AssistantRuntime.",
        "wiki: <query>              -> search knowledge vault trực tiếp",
        "đọc wiki/path/note.md      -> read knowledge note trực tiếp",
        "/k <câu hỏi>               -> ép LLM dùng knowledge context",
        "/trace                     -> bật/tắt trace",
        "/usage                     -> xem token/latency của session",
        "/memory                    -> xem session memory trong RAM",
        "/clear                     -> xóa session memory trong RAM",
        "/exit                      -> thoát",
    ]
)

InputFn = Callable[[str], str]


def run_text_chat(
    config: TextRuntimeConfig,
    *,
    console: Console,
    show_trace: bool = False,
    show_usage: bool = False,
    runtime_builder=build_text_runtime,
    input_fn: InputFn | None = None,
) -> int:
    """Run an interactive text chat session over one shared runtime instance."""
    bundle = runtime_builder(config)
    ask = input_fn or _prompt_user
    session_usage = SessionUsage()

    console.print()
    header = Text()
    header.append(f"{ICON.BIRD} ", style=st(f"bold {ACCENT}"))
    header.append("SoCa", style=st(f"bold {ACCENT}"))
    header.append(" · chat", style=st(MUTED))
    console.print(header)
    console.print(
        Text(
            f"    LLM {bundle.llm_status} {ICON.DOT} knowledge {bundle.knowledge_status}"
            f" {ICON.DOT} memory {bundle.memory_status}",
            style=st(MUTED),
        )
    )
    console.print(Text("    gõ /help để xem lệnh, /exit để thoát", style=st(MUTED)))
    console.print()

    while True:
        try:
            user_text = ask("\nYou").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Chat aborted.[/yellow]")
            return 130

        if not user_text:
            continue
        if user_text in CHAT_EXIT_COMMANDS:
            console.print("[green]Bye.[/green]")
            return 0
        if user_text == "/help":
            console.print(Panel(CHAT_HELP, title=Text("Lệnh chat", style=st(TITLE)), border_style=st(BORDER) or "none", padding=(0, 1)))
            continue
        if user_text == "/trace":
            show_trace = not show_trace
            console.print(f"[yellow]Trace:[/yellow] {'on' if show_trace else 'off'}")
            continue
        if user_text == "/usage":
            render_session_usage(console, session_usage)
            continue
        if user_text == "/clear":
            if bundle.session_memory is None:
                console.print("[yellow]Memory is disabled.[/yellow]")
            else:
                bundle.session_memory.clear()
                console.print("[yellow]Session memory cleared.[/yellow]")
            continue
        if user_text == "/memory":
            console.print(
                Panel(
                    _render_session_memory(bundle),
                    title=Text("Session memory", style=st(TITLE)),
                    border_style=st(BORDER) or "none",
                    padding=(0, 1),
                )
            )
            continue

        normalized_text, metadata = normalize_text_turn(user_text)
        result = bundle.runtime.run_text_turn(
            normalized_text,
            source="cli_chat",
            metadata=metadata,
        )
        render_text_result(console, result, show_trace=show_trace)

        turn_usage = TurnUsage.from_runtime_result(result)
        session_usage = session_usage.add(turn_usage)
        if show_usage:
            print_turn_usage(console, turn_usage)


def _render_session_memory(bundle: TextRuntimeBundle) -> str:
    if bundle.session_memory is None:
        return "Memory is disabled."
    rendered = bundle.session_memory.render().strip()
    return rendered or "<empty>"


def _prompt_user(prompt: str) -> str:
    del prompt
    marker = st(f"bold {ALT}")
    return Prompt.ask(f"[{marker}]{ICON.USER}[/{marker}]" if marker else ICON.USER)
