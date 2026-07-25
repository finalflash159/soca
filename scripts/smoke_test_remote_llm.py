"""Smoke test: call a real remote LLM provider once, end-to-end.

Reads the provider API key from the environment or a local ``.env`` file
(never committed) and runs a single short Vietnamese prompt through
``RemoteOpenAILLM``, printing the answer, real token usage, and latency.

Examples::

    uv run python scripts/smoke_test_remote_llm.py --provider groq
    uv run python scripts/smoke_test_remote_llm.py --all
    uv run python scripts/smoke_test_remote_llm.py --provider openai --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.table import Table

from soca.llm.providers import RemoteLLMError, RemoteOpenAILLM, get_provider
from soca.llm.providers.provider_registry import PROVIDER_REGISTRY

console = Console()


def _mask_key(value: str) -> str:
    """Show only the last 4 characters of an API key (never log the rest)."""
    value = value.strip()
    if len(value) <= 4:
        return "****"
    return f"...{value[-4:]}"

DEFAULT_PROMPT = "Chào bạn, hãy giới thiệu ngắn gọn về bản thân trong một câu."

# Sensible free/cheap default model per provider for a smoke test.
DEFAULT_MODELS = {
    "groq": "llama-3.1-8b-instant",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "openrouter": "meta-llama/llama-3.3-70b-instruct",
}


def load_dotenv(path: Path) -> None:
    """Minimal .env loader (no python-dotenv dependency).

    Only sets variables that are not already present in the environment, so a
    real environment variable always wins over the file.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def run_provider(provider_key: str, model: str | None, max_tokens: int) -> bool:
    provider = get_provider(provider_key)
    api_key = os.environ.get(provider.api_key_env, "").strip()
    model = model or DEFAULT_MODELS[provider_key]

    if not api_key:
        console.print(
            f"[yellow]SKIP[/yellow] {provider.label}: missing {provider.api_key_env}"
        )
        return False

    console.print(
        f"\n[bold]{provider.label}[/bold] · model=[cyan]{model}[/cyan] · "
        f"key=[dim]{_mask_key(api_key)}[/dim]"
    )
    try:
        engine = RemoteOpenAILLM(provider=provider, model=model, api_key=api_key)
        result = engine.generate(DEFAULT_PROMPT, max_tokens=max_tokens)
    except RemoteLLMError as exc:
        console.print(f"  [red]FAIL[/red] ({exc.category}): {exc}")
        return False

    console.print(f"  [green]OK[/green] {result.text}")
    console.print(
        f"  [dim]prompt={result.n_prompt_tokens} tok · "
        f"completion={result.n_completion_tokens} tok · "
        f"{result.total_latency_ms:.0f} ms · {result.tokens_per_second:.1f} tok/s[/dim]"
    )
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remote LLM provider smoke test.")
    parser.add_argument(
        "--provider",
        default="groq",
        choices=sorted(PROVIDER_REGISTRY),
        help="Which provider to call (default: groq, free tier).",
    )
    parser.add_argument("--model", default=None, help="Override the default model id.")
    parser.add_argument("--all", action="store_true", help="Try every provider that has a key.")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens per answer.")
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to a .env file to load keys from (default: .env).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(Path(args.env_file))

    providers = sorted(PROVIDER_REGISTRY) if args.all else [args.provider]
    results: list[tuple[str, bool]] = []
    for key in providers:
        model = None if args.all else args.model
        results.append((key, run_provider(key, model, args.max_tokens)))

    if args.all:
        table = Table(title="Remote LLM smoke summary")
        table.add_column("Provider")
        table.add_column("Result")
        for key, ok in results:
            table.add_row(key, "[green]OK[/green]" if ok else "[red]—[/red]")
        console.print("\n", table)

    any_ok = any(ok for _, ok in results)
    return 0 if any_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
