"""Rendering for runtime usage/telemetry (app layer).

Core (`soca/core/usage.py`) only builds the numbers; this module turns them into
Rich output for `soca ask/chat/voice --usage`. Keeping rendering here means core
never imports Rich.
"""
from __future__ import annotations

from rich.console import Console
from rich.table import Table

from soca.app.style.palette import ALT, st
from soca.core.usage import SessionUsage, TurnUsage


def _ms(value: float | None) -> str:
    return f"{value:.0f}ms" if isinstance(value, (int, float)) else "—"


def format_turn_usage_line(usage: TurnUsage) -> str:
    """One compact line, used after each chat/voice turn."""
    parts = [f"route={usage.route}"]

    if usage.llm is not None:
        llm = usage.llm
        parts.append(f"TTFT {llm.ttft_ms:.0f}ms")
        parts.append(f"{llm.tokens_per_second:.0f} tok/s")
        parts.append(f"prompt {llm.prompt_tokens}")
        parts.append(f"out {llm.completion_tokens}")
    else:
        parts.append("no LLM")

    # Voice-only fields (None on text turns).
    if usage.asr_latency_ms is not None:
        parts.append(f"ASR {_ms(usage.asr_latency_ms)}")
    if usage.ttfa_ms is not None:
        parts.append(f"TTFA {_ms(usage.ttfa_ms)}")
    if usage.tts_chunks is not None:
        parts.append(f"{usage.tts_chunks} chunks")

    if usage.total_turn_latency_ms is not None:
        parts.append(f"total {_ms(usage.total_turn_latency_ms)}")
    elif usage.runtime_latency_ms is not None:
        parts.append(f"llm {_ms(usage.runtime_latency_ms)}")

    return " · ".join(parts)


def print_turn_usage(console: Console, usage: TurnUsage) -> None:
    console.print(f"[dim]usage[/dim] {format_turn_usage_line(usage)}", highlight=False)


def render_session_usage(console: Console, session: SessionUsage) -> None:
    """Session totals, used by `/usage` in chat."""
    table = Table(title="Session Usage")
    table.add_column("Metric", style=st(ALT) or "none")
    table.add_column("Value", justify="right")
    table.add_row("turns", str(session.total_turns))
    table.add_row("LLM turns", str(session.llm_turns))
    table.add_row("prompt tokens", str(session.total_prompt_tokens))
    table.add_row("completion tokens", str(session.total_completion_tokens))
    table.add_row("mean TTFT", f"{session.mean_ttft_ms:.0f} ms")
    table.add_row("mean tok/s", f"{session.mean_tokens_per_second:.1f}")
    console.print(table)


__all__ = ["format_turn_usage_line", "print_turn_usage", "render_session_usage"]
