"""List the real model catalog for each remote provider (uses .env keys).

Fetches the live ``/models`` list for a provider and prints every model with its
context length, price (live for OpenRouter, static table for the others), and the
pricing source. Use it to see full coverage and to spot models missing a price.

Examples::

    PYTHONPATH=. python scripts/list_remote_models.py --provider openai
    PYTHONPATH=. python scripts/list_remote_models.py --all
    PYTHONPATH=. python scripts/list_remote_models.py --provider groq --query llama
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.table import Table

from soca.llm.providers import (
    PRICING_TABLE_AS_OF,
    RemoteLLMError,
    fetch_catalog,
    get_provider,
    search_models,
)
from soca.llm.providers.provider_registry import PROVIDER_REGISTRY

console = Console()


def load_dotenv(path: Path) -> None:
    """Minimal .env loader; real environment variables win over the file."""
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


def _fmt_price(value: float | None) -> str:
    return f"${value:g}" if value is not None else "—"


def _fmt_ctx(value: int | None) -> str:
    if value is None:
        return "—"
    if value >= 1000:
        return f"{value // 1000}k"
    return str(value)


def run_provider(provider_key: str, query: str) -> bool:
    provider = get_provider(provider_key)
    api_key = os.environ.get(provider.api_key_env, "").strip()
    if not api_key:
        console.print(f"[yellow]SKIP[/yellow] {provider.label}: missing {provider.api_key_env}")
        return False

    try:
        catalog = fetch_catalog(provider, api_key)
    except RemoteLLMError as exc:
        console.print(f"[red]FAIL[/red] {provider.label} ({exc.category}): {exc}")
        return False

    shown = search_models(catalog, query)
    priced = sum(1 for m in catalog if m.pricing_source != "unknown")
    src_label = "live" if provider.has_pricing_api else f"table as of {PRICING_TABLE_AS_OF}"

    table = Table(
        title=f"{provider.label} — {len(catalog)} models "
        f"({priced} priced · {src_label}) · showing {len(shown)}"
    )
    table.add_column("id", style="cyan", no_wrap=True)
    table.add_column("ctx", justify="right")
    table.add_column("in/1M", justify="right")
    table.add_column("out/1M", justify="right")
    table.add_column("src")
    for m in sorted(shown, key=lambda x: x.id):
        table.add_row(
            m.id,
            _fmt_ctx(m.context_length),
            _fmt_price(m.price_prompt_per_1m),
            _fmt_price(m.price_completion_per_1m),
            m.pricing_source,
        )
    console.print(table)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List remote model catalogs.")
    parser.add_argument("--provider", default="openai", choices=sorted(PROVIDER_REGISTRY))
    parser.add_argument("--all", action="store_true", help="List every provider that has a key.")
    parser.add_argument("--query", default="", help="Filter models by keyword.")
    parser.add_argument("--env-file", default=".env", help="Path to a .env file (default: .env).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(Path(args.env_file))

    providers = sorted(PROVIDER_REGISTRY) if args.all else [args.provider]
    any_ok = False
    for key in providers:
        any_ok = run_provider(key, args.query) or any_ok
    return 0 if any_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
