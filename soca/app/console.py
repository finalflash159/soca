"""Console renderers for `soca voice` — dawn palette, product-grade output.

Per-chunk engineering events (audio/tts) stay hidden behind
``SOCA_VERBOSE_EVENTS=1``; the default output reads like a conversation:

    (o> SoCa · voice
        baseline · ASR phowhisper · LLM qwen3 · TTS valtec/NF

    ● Đang nghe… (nói tự nhiên, im lặng để kết thúc)
    ❯ xin chào sơn ca
    (o> Chào bạn! ...
      · llm · 2.1s · TTFA 210ms
"""

from __future__ import annotations

import os
from typing import Any

from rich.console import Console
from rich.text import Text

from soca.app.style.palette import (
    ACCENT,
    ALT,
    BAD,
    BORDER,
    GOOD,
    ICON,
    MUTED,
    TEXT,
    WARN,
    st,
)
from soca.core.streaming import StreamingEvent
from soca.core.voice_runtime import VoiceRuntimeWarmupResult

VERBOSE_EVENTS = os.environ.get("SOCA_VERBOSE_EVENTS", "") not in ("", "0")


def _line(*spans: tuple[str, str]) -> Text:
    text = Text()
    for content, style in spans:
        text.append(content, style=st(style))
    return text


def print_runtime_header(
    console: Console,
    *,
    profile_key: str,
    asr_model: str,
    llm_model: str,
    tts_model: str,
    voice: str | None,
    memory_status: str,
    knowledge_status: str,
    asr_guard_status: str,
) -> None:
    """Print the resolved voice runtime before entering the recording loop."""
    console.print()
    console.print(_line((f"{ICON.BIRD} ", f"bold {ACCENT}"), ("SoCa", f"bold {ACCENT}"), (" · voice", MUTED)))
    console.print(
        _line(
            ("    ", ""),
            (profile_key, ALT),
            (f" {ICON.DOT} ASR {asr_model} {ICON.DOT} LLM {llm_model} {ICON.DOT} TTS {tts_model}", MUTED),
            (f"/{voice}" if voice else "", MUTED),
        )
    )
    console.print(
        _line(
            ("    ", ""),
            (f"memory {memory_status} {ICON.DOT} knowledge {knowledge_status} {ICON.DOT} guards {asr_guard_status}", MUTED),
        )
    )
    console.print()


def print_waiting_for_speech(console: Console, *, manual_start: bool = False) -> None:
    hint = "nói tự nhiên, im lặng để kết thúc" if not manual_start else "nói đi, ngừng nói để kết thúc lượt"
    console.print(
        _line((f"{ICON.STATE_ON} ", ACCENT), ("Đang nghe… ", f"bold {ACCENT}"), (f"({hint})", MUTED))
    )


def print_recorded_audio(console: Console, *, duration_s: float) -> None:
    console.print(_line((f"{ICON.DOT} ", MUTED), (f"ghi {duration_s:.1f}s, đang xử lý…", MUTED)))


def print_warmup_start(console: Console) -> None:
    console.print(_line((f"{ICON.STATE_HALF} ", WARN), ("khởi động runtime…", MUTED)))


def print_warmup_result(console: Console, result: VoiceRuntimeWarmupResult) -> None:
    if result.ok:
        console.print(
            _line(
                (f"{ICON.OK} ", GOOD),
                (f"{result.component} ", TEXT),
                (f"{result.latency_ms:.0f}ms", MUTED),
            )
        )
        return
    console.print(
        _line(
            (f"{ICON.ERR} ", BAD),
            (f"{result.component} lỗi ", BAD),
            (f"({result.latency_ms:.0f}ms) {result.detail}", MUTED),
        )
    )


def print_followup(console: Console, text: str) -> None:
    """Render a conversation-repair follow-up (not an error)."""
    console.print(_line(("Follow-up: ", f"bold {WARN}"), (text, TEXT)))


def print_streaming_event(console: Console, event: StreamingEvent) -> None:
    """Render a VoicePipeline streaming event without performing side effects."""
    if event.type == "asr":
        console.print(
            _line((f"{ICON.USER} ", f"bold {ALT}"), (event.text or "<trống>", TEXT))
        )
    elif event.type == "repair":
        kind = _metadata_value(event.metadata, "repair_kind", "")
        console.print(
            _line(
                ("Follow-up: ", f"bold {WARN}"),
                (event.text or "", TEXT),
                (f" ({kind})" if kind else "", MUTED),
            )
        )
    elif event.type == "runtime":
        # Only reached when tokens did NOT stream live (voice_loop skips it otherwise).
        console.print(
            _line((f"{ICON.BIRD} ", f"bold {ACCENT}"), (event.text or "<trống>", TEXT))
        )
    elif event.type == "audio":
        if VERBOSE_EVENTS:
            console.print(_line((f"{ICON.DOT} chunk: {event.text}", MUTED)))
    elif event.type == "tts":
        if VERBOSE_EVENTS:
            ttfa_ms = _metadata_value(event.metadata, "ttfa_ms")
            suffix = f" (TTFA {ttfa_ms:.0f}ms)" if isinstance(ttfa_ms, (int, float)) else ""
            console.print(_line((f"{ICON.DOT} nói: {event.text}{suffix}", MUTED)))
    elif event.type == "error":
        console.print(_line((f"{ICON.ERR} ", f"bold {BAD}"), (event.text or "lỗi", BAD)))
    elif event.type == "done":
        parts: list[str] = []
        route = _metadata_value(event.metadata, "runtime_route", "")
        if route:
            parts.append(str(route))
        if event.latency_ms is not None:
            parts.append(f"{event.latency_ms / 1000:.1f}s")
        ttfa = _metadata_value(event.metadata, "ttfa_ms")
        if isinstance(ttfa, (int, float)):
            parts.append(f"TTFA {ttfa:.0f}ms")
        summary = f" {ICON.DOT} ".join(parts) if parts else "xong"
        console.print(_line((f"  {ICON.DOT} ", MUTED), (summary, MUTED)))
        console.print(_line((ICON.RULE * 42, BORDER)))


def _metadata_value(metadata: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    if metadata is None:
        return default
    return metadata.get(key, default)
