"""Smoke test: load a local llama.cpp LLM and run 5 Vietnamese prompts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from rich.console import Console
from rich.table import Table

from soca.llm import LocalLlamaCppLLM
from soca.llm.registry import DEFAULT_LLM_MODEL_KEY, LLM_MODEL_REGISTRY

console = Console()

TEST_PROMPTS = [
    "Thủ đô của Việt Nam là gì?",
    "Hãy giới thiệu ngắn về bản thân bạn.",
    "Mấy giờ rồi?",
    "Làm sao để đặt một báo thức?",
    "Kể cho tôi một câu chuyện cười ngắn về lập trình viên.",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a quick SoCa LLM smoke test.")
    parser.add_argument(
        "--model",
        default=DEFAULT_LLM_MODEL_KEY,
        choices=sorted(LLM_MODEL_REGISTRY),
        help="LLM registry key to load.",
    )
    parser.add_argument("--max-tokens", type=int, default=120, help="Maximum tokens per prompt.")
    parser.add_argument("--n-threads", type=int, default=8, help="llama.cpp CPU threads.")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="llama.cpp GPU layers.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    console.print(f"[bold]Loading llama.cpp LLM:[/bold] {args.model}")
    llm = LocalLlamaCppLLM(
        model_key=args.model,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
    )
    console.print("[green]✓ Model loaded[/green]\n")

    # Warm-up: first call is often slower because llama.cpp initializes KV/cache paths.
    console.print("[dim]Warming up...[/dim]")
    _ = llm.generate("Xin chào", max_tokens=20)
    console.print("[dim]Warm-up done.[/dim]\n")

    table = Table(title=f"LLM Smoke Test — {args.model}", show_lines=True)
    table.add_column("Prompt", style="cyan", width=30)
    table.add_column("Response", style="white", width=80, overflow="fold")
    table.add_column("TTFT (ms)", justify="right", style="yellow")
    table.add_column("Total (ms)", justify="right", style="yellow")
    table.add_column("tok/s", justify="right", style="magenta")
    table.add_column("# tok", justify="right")

    for prompt in TEST_PROMPTS:
        result = llm.generate(prompt, max_tokens=args.max_tokens, temperature=0.7)
        table.add_row(
            prompt,
            result.text,
            f"{result.ttft_ms:.0f}",
            f"{result.total_latency_ms:.0f}",
            f"{result.tokens_per_second:.1f}",
            str(result.n_completion_tokens),
        )

    console.print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
