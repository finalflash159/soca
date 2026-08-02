from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from soca.config import LlmSettings
from soca.core import AssistantRuntime, RuntimeOptions
from soca.llm.factory import DEFAULT_LLM_ENGINE_FACTORY
from soca.llm.providers import RemoteLLMError, get_provider
from soca.llm.providers.provider_registry import PROVIDER_REGISTRY

console = Console()
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-3.5-flash-lite",
    "openrouter": "google/gemini-2.5-flash-lite",
    "groq": "llama-3.1-8b-instant",
}
SURFACE_PROMPTS = {
    "chat": "Trả lời đúng một câu tiếng Việt: retrieval khác generation ở điểm nào?",
    "voice_transcript": "Giải thích ngắn gọn bằng tiếng Việt: working memory dùng để làm gì?",
}


class EnvironmentSecrets:
    def get_key(self, provider_key: str) -> str | None:
        provider = get_provider(provider_key)
        return os.environ.get(provider.api_key_env, "").strip() or None


@dataclass(frozen=True)
class SmokeReceipt:
    provider: str
    model: str
    surface: str
    provider_called: bool
    route: str
    terminal: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    response: str
    provider_trace: dict[str, Any]
    error: dict[str, Any]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def _model_for(provider_key: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    override = os.environ.get(f"SOCA_SMOKE_{provider_key.upper()}_MODEL", "").strip()
    return override or DEFAULT_MODELS[provider_key]


def run_provider(
    provider_key: str,
    model: str,
    *,
    max_tokens: int,
) -> list[SmokeReceipt]:
    provider = get_provider(provider_key)
    secrets = EnvironmentSecrets()
    if secrets.get_key(provider_key) is None:
        console.print(f"[yellow]SKIP[/yellow] {provider.label}: missing {provider.api_key_env}")
        return []

    settings = LlmSettings(
        backend="remote",
        provider_key=provider_key,
        model_id=model,
        max_tokens=2_048,
    )
    engine = DEFAULT_LLM_ENGINE_FACTORY(settings, secrets)
    runtime = AssistantRuntime(
        llm=engine,
        options=RuntimeOptions(max_tokens=max_tokens, model_max_output_tokens=2_048),
    )
    receipts: list[SmokeReceipt] = []
    console.print(f"\n[bold]{provider.label}[/bold] · [cyan]{model}[/cyan]")
    for surface, prompt in SURFACE_PROMPTS.items():
        source = "voice" if surface == "voice_transcript" else "text"
        try:
            result = runtime.run_text_turn(prompt, source=source)
            trace = result.trace
            provider_trace = dict(trace.provider_trace) if trace is not None else {}
            llm_error = dict(trace.llm_error) if trace is not None else {}
            usage = result.usage
            terminal = "safe_failure" if result.blocked else "achieved"
            receipt = SmokeReceipt(
                provider=provider_key,
                model=model,
                surface=surface,
                provider_called=bool(provider_trace),
                route=result.route.value,
                terminal=terminal,
                prompt_tokens=usage.prompt_tokens if usage is not None else 0,
                completion_tokens=usage.completion_tokens if usage is not None else 0,
                latency_ms=usage.total_latency_ms if usage is not None else 0.0,
                response=result.response_text,
                provider_trace=provider_trace,
                error=llm_error,
            )
        except RemoteLLMError as exc:
            receipt = SmokeReceipt(
                provider=provider_key,
                model=model,
                surface=surface,
                provider_called=exc.attempts > 0,
                route="provider_error",
                terminal="system_failure",
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=0.0,
                response="",
                provider_trace={},
                error=exc.as_dict(),
            )
        receipts.append(receipt)
        color = "green" if receipt.terminal == "achieved" else "red"
        console.print(
            f"  [{color}]{receipt.terminal.upper()}[/{color}] {surface} · "
            f"{receipt.latency_ms:.0f} ms · {receipt.completion_tokens} output tok"
        )
    return receipts


def _write_artifact(path: Path, receipts: list[SmokeReceipt]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "voice_scope": "transcript-only; no microphone, ASR, TTS, or audio hardware",
        "receipts": [asdict(receipt) for receipt in receipts],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real remote provider chat/transcript smoke.")
    parser.add_argument("--provider", choices=sorted(PROVIDER_REGISTRY), default="openrouter")
    parser.add_argument("--model")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--artifact", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 16 <= args.max_tokens <= 2_048:
        raise SystemExit("--max-tokens must be from 16 to 2048")
    load_dotenv(Path(args.env_file))
    provider_keys = sorted(PROVIDER_REGISTRY) if args.all else [args.provider]
    receipts: list[SmokeReceipt] = []
    for provider_key in provider_keys:
        model = _model_for(provider_key, None if args.all else args.model)
        receipts.extend(run_provider(provider_key, model, max_tokens=args.max_tokens))

    if args.artifact is not None:
        _write_artifact(args.artifact, receipts)
        console.print(f"\nartifact: {args.artifact}")

    table = Table(title="Remote provider smoke")
    table.add_column("Provider")
    table.add_column("Receipts")
    table.add_column("Result")
    for provider_key in provider_keys:
        provider_receipts = [item for item in receipts if item.provider == provider_key]
        passed = len(provider_receipts) == len(SURFACE_PROMPTS) and all(
            item.terminal == "achieved" and item.provider_called for item in provider_receipts
        )
        table.add_row(provider_key, str(len(provider_receipts)), "PASS" if passed else "FAIL/SKIP")
    console.print(table)

    passed_providers = {
        provider_key
        for provider_key in provider_keys
        if len([item for item in receipts if item.provider == provider_key]) == len(SURFACE_PROMPTS)
        and all(
            item.terminal == "achieved" and item.provider_called
            for item in receipts
            if item.provider == provider_key
        )
    }
    if args.require_all:
        return 0 if passed_providers == set(provider_keys) else 1
    return 0 if passed_providers else 1


if __name__ == "__main__":
    raise SystemExit(main())
