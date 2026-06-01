from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from shrike7.asr import ASR_MODEL_REGISTRY
from shrike7.core import (
    DEFAULT_VOICE_RUNTIME_PROFILE_KEY,
    VOICE_RUNTIME_PROFILES,
    EndpointConfig,
    SoundDevicePlayer,
    build_voice_runtime,
    record_until_silence,
    resolve_voice_runtime_config,
)
from shrike7.llm.registry import LLM_MODEL_REGISTRY
from shrike7.tts import TTS_MODEL_REGISTRY

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local Shrike-7 voice loop.")
    parser.add_argument(
        "--profile",
        default=DEFAULT_VOICE_RUNTIME_PROFILE_KEY,
        choices=sorted(VOICE_RUNTIME_PROFILES),
        help="Voice runtime profile to use before explicit model overrides.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        choices=sorted(LLM_MODEL_REGISTRY),
        help="Override the LLM registry key from the selected profile.",
    )
    parser.add_argument(
        "--asr-model",
        default=None,
        choices=sorted(ASR_MODEL_REGISTRY),
        help="Override the ASR registry key from the selected profile.",
    )
    parser.add_argument(
        "--tts-model",
        default=None,
        choices=sorted(TTS_MODEL_REGISTRY),
        help="Override the TTS registry key from the selected profile.",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help="Override the TTS voice/speaker id from the selected profile.",
    )
    parser.add_argument("--endpoint-silence-ms", type=int, default=None)
    parser.add_argument("--max-record-ms", type=int, default=None)
    parser.add_argument(
        "--vault",
        type=Path,
        default=Path.home() / "KnowledgeVault",
        help="Knowledge vault root containing memory/profile.md.",
    )
    parser.add_argument("--no-memory", action="store_true", help="Disable profile/session memory.")
    parser.add_argument("--memory-chars", type=int, default=2200)
    parser.add_argument("--profile-chars", type=int, default=900)
    parser.add_argument("--session-chars", type=int, default=1300)
    parser.add_argument("--session-turns", type=int, default=6)
    parser.add_argument("--turn-chars", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--no-speak-rejections",
        action="store_true",
        help="Do not speak the fallback response when ASR rejects a turn.",
    )
    return parser


def resolve_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    config = resolve_voice_runtime_config(
        profile_key=args.profile,
        asr_model=args.asr_model,
        llm_model=args.llm_model,
        tts_model=args.tts_model,
        tts_voice=args.voice,
        endpoint_silence_ms=args.endpoint_silence_ms,
        max_record_ms=args.max_record_ms,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        vault=args.vault,
        no_memory=args.no_memory,
        memory_chars=args.memory_chars,
        profile_chars=args.profile_chars,
        session_chars=args.session_chars,
        session_turns=args.session_turns,
        turn_chars=args.turn_chars,
    )
    args.asr_model = config.asr_model
    args.llm_model = config.llm_model
    args.tts_model = config.tts_model
    args.voice = config.tts_voice
    args.endpoint_silence_ms = config.endpoint_silence_ms
    args.max_record_ms = config.max_record_ms
    args.max_tokens = config.max_tokens
    args.temperature = config.temperature
    args.top_p = config.top_p
    args.vault = config.vault
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = resolve_runtime_args(build_parser().parse_args(argv))
    runtime_config = resolve_voice_runtime_config(
        profile_key=args.profile,
        asr_model=args.asr_model,
        llm_model=args.llm_model,
        tts_model=args.tts_model,
        tts_voice=args.voice,
        endpoint_silence_ms=args.endpoint_silence_ms,
        max_record_ms=args.max_record_ms,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        vault=args.vault,
        no_memory=args.no_memory,
        memory_chars=args.memory_chars,
        profile_chars=args.profile_chars,
        session_chars=args.session_chars,
        session_turns=args.session_turns,
        turn_chars=args.turn_chars,
    )
    bundle = build_voice_runtime(runtime_config)
    detector = bundle.detector
    tts = bundle.tts
    player = SoundDevicePlayer()

    pipeline = bundle.pipeline
    endpoint_config = EndpointConfig(
        endpoint_silence_ms=args.endpoint_silence_ms,
        max_record_ms=args.max_record_ms,
    )

    console.print(
        f"[green]Voice loop ready[/green] profile={args.profile} ASR={args.asr_model} "
        f"LLM={args.llm_model} TTS={args.tts_model} voice={args.voice}"
    )
    console.print(f"[dim]Memory:[/dim] {bundle.memory_status}")
    console.print(f"[dim]Knowledge:[/dim] {bundle.knowledge_status}")
    console.print(f"[dim]ASR guards:[/dim] {bundle.asr_guard_status}")

    while True:
        input("\nPress ENTER and speak. Ctrl+C to quit.")
        console.print("[cyan]Recording...[/cyan] Speak now. Stop talking to end the turn.")
        audio = record_until_silence(detector, config=endpoint_config)
        duration_s = (
            len(audio) / endpoint_config.sample_rate if endpoint_config.sample_rate > 0 else 0.0
        )
        console.print(f"[dim]Recorded {duration_s:.2f}s. Processing...[/dim]")

        for event in pipeline.turn_streaming(audio, audio_sink=player):
            if event.type == "asr":
                console.print(Panel(event.text or "<empty>", title="ASR"))
            elif event.type == "runtime":
                route = event.metadata.get("route") if event.metadata else ""
                title = f"Runtime: {route}" if route else "Runtime"
                console.print(Panel(event.text or "<empty>", title=title, border_style="blue"))
            elif event.type == "audio":
                console.print(f"[dim]Played chunk:[/dim] {event.text}")
            elif event.type == "tts":
                ttfa_ms = event.metadata.get("ttfa_ms") if event.metadata else None
                suffix = f" ({ttfa_ms:.0f} ms TTFA)" if ttfa_ms is not None else ""
                console.print(f"[green]Speaking chunk:[/green] {event.text}{suffix}")
            elif event.type == "error":
                console.print(f"[red]Streaming error:[/red] {event.text}")
            elif event.type == "done":
                rejected = bool(event.metadata and event.metadata.get("rejected"))
                if rejected and event.text and not args.no_speak_rejections:
                    console.print(f"[yellow]Fallback:[/yellow] {event.text}")
                    tts_result = tts.synthesize(event.text)
                    player.play(tts_result.audio, tts_result.sample_rate, blocking=True)
                console.print(f"\n[dim]Done in {event.latency_ms:.0f} ms[/dim]")


if __name__ == "__main__":
    raise SystemExit(main())
