from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

from rich.console import Console

from soca.llm import LocalLlamaCppLLM
from soca.llm.registry import DEFAULT_LLM_MODEL_KEY, LLM_MODEL_REGISTRY

console = Console()
PROMPT = "Giải thích ngắn gọn về trí tuệ nhân tạo cho tôi học sinh cấp 2."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Demo token streaming from a local LLM.")
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL_KEY,
        choices=sorted(LLM_MODEL_REGISTRY),
        help="LLM registry key to load.",
    )
    parser.add_argument("--prompt", default=PROMPT, help="Prompt to stream.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    llm = LocalLlamaCppLLM(model_key=args.llm_model, n_threads=8, n_gpu_layers=-1)
    console.print(f"[green]✓ Model ready:[/green] {args.llm_model}\n")

    console.print(f"[bold cyan]Câu hỏi: [/bold cyan] {args.prompt}\n")
    console.print("[bold yellow]Sơn Ca: [/bold yellow] ", end="")

    t0 = time.perf_counter()
    first_token_time = None
    n_tokens = 0

    for chunk in llm.generate_stream(args.prompt, max_tokens=200, temperature=0.7):
        if first_token_time is None:
            first_token_time = time.perf_counter()

        print(chunk, end="", flush=True)
        n_tokens += 1

    t1 = time.perf_counter()
    if first_token_time is None:
        first_token_time = t1
    ttft = (first_token_time - t0) * 1000
    total_latency = (t1 - t0) * 1000
    gen = (t1 - first_token_time) * 1000

    console.print(
        f"\n\n[dim]TTFT: {ttft:.0f} ms | Total latency: {total_latency:.0f} ms | "
        f"Gen time: {gen:.0f} ms | Tokens: {n_tokens}[/dim]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
