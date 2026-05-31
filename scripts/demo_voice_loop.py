from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from shrike7.asr import ASR_MODEL_REGISTRY, SpeechDetector
from shrike7.asr.robust_asr import RobustASR
from shrike7.asr.whisper_onnx import VietnameseASR
from shrike7.core import (
    DEFAULT_VOICE_RUNTIME_PROFILE_KEY,
    VOICE_RUNTIME_PROFILES,
    AssistantRuntime,
    DefaultRuntimeToolRouter,
    EndpointConfig,
    RuntimeOptions,
    SoundDevicePlayer,
    get_voice_runtime_profile,
    record_until_silence,
)
from shrike7.core.pipeline import VoicePipeline
from shrike7.knowledge import KnowledgeContextBuilder, MarkdownVaultKnowledgeSource
from shrike7.llm import LocalLlamaCppLLM
from shrike7.llm.registry import LLM_MODEL_REGISTRY
from shrike7.memory import MarkdownLongTermMemory, MemoryContextBuilder, SessionMemory
from shrike7.tools import (
    KnowledgeReadTool,
    KnowledgeSearchTool,
    LocalTimerTool,
    LocalTimeTool,
    ToolRuntime,
)
from shrike7.tts import TTS_MODEL_REGISTRY, create_tts_engine

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
    profile = get_voice_runtime_profile(args.profile)
    resolved_tts_model = args.tts_model or profile.tts_model

    args.asr_model = args.asr_model or profile.asr_model
    args.llm_model = args.llm_model or profile.llm_model
    args.tts_model = resolved_tts_model

    if args.voice is not None:
        args.voice = args.voice
    elif resolved_tts_model == profile.tts_model:
        args.voice = profile.tts_voice or TTS_MODEL_REGISTRY[resolved_tts_model].default_voice
    else:
        args.voice = TTS_MODEL_REGISTRY[resolved_tts_model].default_voice

    args.endpoint_silence_ms = (
        args.endpoint_silence_ms
        if args.endpoint_silence_ms is not None
        else profile.endpoint_silence_ms
    )
    args.max_record_ms = args.max_record_ms if args.max_record_ms is not None else profile.max_record_ms
    args.max_tokens = args.max_tokens if args.max_tokens is not None else profile.max_tokens
    args.temperature = args.temperature if args.temperature is not None else profile.temperature
    args.top_p = args.top_p if args.top_p is not None else profile.top_p
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = resolve_runtime_args(build_parser().parse_args(argv))
    args.vault = args.vault.expanduser().resolve()

    detector = SpeechDetector()
    asr = RobustASR(asr=VietnameseASR(model_key=args.asr_model), vad=detector)
    base_llm = LocalLlamaCppLLM(model_key=args.llm_model, n_threads=8, n_gpu_layers=-1)

    memory_status = "disabled"
    knowledge_status = "disabled"
    memory_builder = None
    knowledge_builder = None
    tools = [LocalTimeTool(), LocalTimerTool()]

    if args.vault.is_dir():
        source = MarkdownVaultKnowledgeSource(args.vault, include_globs=("wiki/**/*.md",))
        knowledge_builder = KnowledgeContextBuilder(source)
        tools.extend([KnowledgeSearchTool(source), KnowledgeReadTool(source)])
        knowledge_status = f"enabled:{args.vault / 'wiki'}"
    else:
        knowledge_status = f"disabled:not_found:{args.vault}"
        if not args.no_memory:
            memory_status = f"disabled:not_found:{args.vault}"

    if not args.no_memory and args.vault.is_dir():
        long_term_memory = MarkdownLongTermMemory(
            args.vault,
            max_chars=args.profile_chars,
        )
        session_memory = SessionMemory(
            max_turns=args.session_turns,
            max_chars=args.session_chars,
            max_turn_chars=args.turn_chars,
        )
        memory_builder = MemoryContextBuilder(
            long_term=long_term_memory,
            session=session_memory,
            max_chars=args.memory_chars,
            profile_chars=args.profile_chars,
        )
        memory_status = f"enabled:{args.vault / 'memory' / 'profile.md'}"

    tool_runtime = ToolRuntime(tools)
    assistant_runtime = AssistantRuntime(
        llm=base_llm,
        tool_runtime=tool_runtime,
        tool_router=DefaultRuntimeToolRouter(
            knowledge_search_prefixes=("wiki:", "knowledge:", "wiki ", "knowledge ")
        ),
        knowledge_builder=knowledge_builder,
        memory_builder=memory_builder,
        options=RuntimeOptions(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        ),
    )

    tts_voice = args.voice
    tts = create_tts_engine(args.tts_model, voice=tts_voice, lazy=False)
    player = SoundDevicePlayer()

    pipeline = VoicePipeline(asr=asr, llm=base_llm, tts=tts, assistant_runtime=assistant_runtime)
    config = EndpointConfig(
        endpoint_silence_ms=args.endpoint_silence_ms,
        max_record_ms=args.max_record_ms,
    )

    console.print(
        f"[green]Voice loop ready[/green] profile={args.profile} ASR={args.asr_model} "
        f"LLM={args.llm_model} TTS={args.tts_model} voice={tts_voice}"
    )
    console.print(f"[dim]Memory:[/dim] {memory_status}")
    console.print(f"[dim]Knowledge:[/dim] {knowledge_status}")
    console.print(
        f"[dim]ASR guards:[/dim] BoH={asr.boh_status}; "
        f"confidence={asr.confidence_guard_status}"
    )

    while True:
        input("\nPress ENTER and speak. Ctrl+C to quit.")
        console.print("[cyan]Recording...[/cyan] Speak now. Stop talking to end the turn.")
        audio = record_until_silence(detector, config=config)
        duration_s = len(audio) / config.sample_rate if config.sample_rate > 0 else 0.0
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
