from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel

from soca.core.streaming import StreamingEvent
from soca.core.voice_runtime import VoiceRuntimeWarmupResult


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
    console.print(
        f"[green]Sơn Ca ready[/green] profile={profile_key} ASR={asr_model} "
        f"LLM={llm_model} TTS={tts_model} voice={voice}"
    )
    console.print(f"[dim]Memory:[/dim] {memory_status}")
    console.print(f"[dim]Knowledge:[/dim] {knowledge_status}")
    console.print(f"[dim]ASR guards:[/dim] {asr_guard_status}")


def print_waiting_for_speech(console: Console, *, manual_start: bool = False) -> None:
    if manual_start:
        console.print("[cyan]Recording...[/cyan] Speak now. Stop talking to end the turn.")
        return

    console.print("[cyan]Listening...[/cyan] Speak now. Stop talking to end the turn.")


def print_recorded_audio(console: Console, *, duration_s: float) -> None:
    console.print(f"[dim]Recorded {duration_s:.2f}s. Processing...[/dim]")


def print_warmup_start(console: Console) -> None:
    console.print("[cyan]Warming up runtime...[/cyan]")


def print_warmup_result(console: Console, result: VoiceRuntimeWarmupResult) -> None:
    status = "ok" if result.ok else "failed"
    style = "green" if result.ok else "yellow"
    console.print(
        f"[{style}]Warmup {result.component}:[/{style}] {status} "
        f"({result.latency_ms:.0f} ms) [dim]{result.detail}[/dim]"
    )


def print_rejection_fallback(console: Console, text: str) -> None:
    console.print(f"[yellow]Fallback:[/yellow] {text}")


def print_streaming_event(console: Console, event: StreamingEvent) -> None:
    """Render a VoicePipeline streaming event without performing side effects."""
    if event.type == "asr":
        console.print(Panel(event.text or "<empty>", title="ASR"))
    elif event.type == "runtime":
        route = _metadata_value(event.metadata, "route", "")
        title = f"Runtime: {route}" if route else "Runtime"
        console.print(Panel(event.text or "<empty>", title=title, border_style="blue"))
    elif event.type == "audio":
        console.print(f"[dim]Played chunk:[/dim] {event.text}")
    elif event.type == "tts":
        ttfa_ms = _metadata_value(event.metadata, "ttfa_ms")
        tts_latency_ms = _metadata_value(event.metadata, "tts_latency_ms")
        if isinstance(ttfa_ms, (int, float)):
            suffix = f" ({ttfa_ms:.0f} ms TTFA)"
        elif isinstance(tts_latency_ms, (int, float)):
            suffix = f" ({tts_latency_ms:.0f} ms TTS)"
        else:
            suffix = ""
        console.print(f"[green]Speaking chunk:[/green] {event.text}{suffix}")
    elif event.type == "error":
        console.print(f"[red]Streaming error:[/red] {event.text}")
    elif event.type == "done":
        latency = f"{event.latency_ms:.0f} ms" if event.latency_ms is not None else "n/a"
        console.print(f"\n[dim]Done in {latency}[/dim]")


def _metadata_value(metadata: dict[str, Any] | None, key: str, default: Any = None) -> Any:
    if metadata is None:
        return default
    return metadata.get(key, default)
