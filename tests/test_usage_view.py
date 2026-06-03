from __future__ import annotations

from rich.console import Console

from soca.app.usage_view import (
    format_turn_usage_line,
    print_turn_usage,
    render_session_usage,
)
from soca.core.usage import LLMUsage, SessionUsage, TurnUsage


def _llm() -> LLMUsage:
    return LLMUsage(
        prompt_tokens=142,
        completion_tokens=38,
        ttft_ms=78.0,
        total_latency_ms=1200.0,
        tokens_per_second=62.0,
    )


def test_format_text_turn_line_has_token_metrics() -> None:
    line = format_turn_usage_line(
        TurnUsage(route="free_chat", used_llm=True, llm=_llm(), runtime_latency_ms=1200.0)
    )
    assert "route=free_chat" in line
    assert "TTFT 78ms" in line
    assert "62 tok/s" in line
    assert "prompt 142" in line
    assert "out 38" in line


def test_format_tool_turn_line_marks_no_llm() -> None:
    line = format_turn_usage_line(
        TurnUsage(route="tool_direct", used_tool=True, runtime_latency_ms=3.0)
    )
    assert "route=tool_direct" in line
    assert "no LLM" in line


def test_format_voice_turn_line_has_audio_metrics() -> None:
    line = format_turn_usage_line(
        TurnUsage(
            route="free_chat",
            used_llm=True,
            llm=_llm(),
            asr_latency_ms=480.0,
            ttfa_ms=410.0,
            tts_chunks=2,
            total_turn_latency_ms=1400.0,
        )
    )
    assert "ASR 480ms" in line
    assert "TTFA 410ms" in line
    assert "2 chunks" in line
    assert "total 1400ms" in line


def test_print_turn_usage_does_not_crash() -> None:
    console = Console(record=True, width=120)
    print_turn_usage(console, TurnUsage(route="free_chat", used_llm=True, llm=_llm()))
    assert "route=free_chat" in console.export_text()


def test_render_session_usage_table() -> None:
    console = Console(record=True, width=80)
    session = (
        SessionUsage()
        .add(TurnUsage(route="free_chat", used_llm=True, llm=_llm()))
        .add(TurnUsage(route="tool_direct", used_tool=True))
    )

    render_session_usage(console, session)
    out = console.export_text()

    assert "Session Usage" in out
    assert "prompt tokens" in out
    assert "142" in out  # one llm turn's prompt tokens
