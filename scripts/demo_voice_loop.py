from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from soca.app import run_voice_loop
from soca.asr import ASR_MODEL_REGISTRY
from soca.core import (
    DEFAULT_VOICE_RUNTIME_PROFILE_KEY,
    VOICE_RUNTIME_PROFILES,
    ResolvedVoiceRuntimeConfig,
    resolve_voice_runtime_config,
)
from soca.llm.registry import LLM_MODEL_REGISTRY


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local SoCa voice loop.")
    parser.add_argument(
        "--profile",
        default=DEFAULT_VOICE_RUNTIME_PROFILE_KEY,
        choices=sorted(VOICE_RUNTIME_PROFILES),
        help="Voice runtime profile to use before ASR/LLM/voice overrides.",
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
        "--voice",
        default=None,
        help="Override the Valtec voice/speaker id from the selected profile.",
    )
    parser.add_argument("--endpoint-silence-ms", type=int, default=None)
    parser.add_argument("--max-record-ms", type=int, default=None)
    parser.add_argument(
        "--vault",
        type=Path,
        default=Path.home() / "KnowledgeVault",
        help="Knowledge vault root containing memory/profile.md.",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable profile/session memory.",
    )
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
    parser.add_argument(
        "--press-enter-to-record",
        action="store_true",
        help="Wait for ENTER before each recorded turn. Useful for debugging.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip ASR/LLM/TTS first-call warmup before listening.",
    )
    return parser


def build_runtime_config(args: argparse.Namespace) -> ResolvedVoiceRuntimeConfig:
    return resolve_voice_runtime_config(
        profile_key=args.profile,
        asr_model=args.asr_model,
        llm_model=args.llm_model,
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


def resolve_runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    config = build_runtime_config(args)
    args.asr_model = config.asr_model
    args.llm_model = config.llm_model
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
    print(
        "Note: scripts/demo_voice_loop.py is kept for compatibility. Prefer: uv run soca voice",
        file=sys.stderr,
    )
    return run_voice_loop(
        build_runtime_config(args),
        no_speak_rejections=args.no_speak_rejections,
        press_enter_to_record=args.press_enter_to_record,
        warmup=not args.no_warmup,
    )


if __name__ == "__main__":
    raise SystemExit(main())
