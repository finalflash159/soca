from __future__ import annotations

import argparse
from pathlib import Path

import onnxruntime as ort
from rich.console import Console
from rich.table import Table

from soca.asr import (
    ASR_MODEL_REGISTRY,
    DEFAULT_ASR_MODEL_KEY,
    VietnameseASR,
    get_asr_model_config,
)

console = Console()


def providers_from_arg(value: str) -> list[str]:
    if value == "cpu":
        return ["CPUExecutionProvider"]
    providers = ort.get_available_providers()
    if "CoreMLExecutionProvider" in providers:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local PhoWhisper ONNX smoke test.")
    parser.add_argument(
        "--model",
        choices=sorted(ASR_MODEL_REGISTRY),
        default=DEFAULT_ASR_MODEL_KEY,
        help="ASR model key.",
    )
    parser.add_argument(
        "--providers",
        choices=["auto", "cpu"],
        default="auto",
        help="ONNX Runtime provider policy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = get_asr_model_config(args.model)
    test_dir = Path(__file__).parent.parent / "data" / "test_audio"
    test_files = sorted(test_dir.glob("sample_*.wav"))

    if not test_files:
        console.print(
            "[red]No test audio. Please run:[/red] "
            "[green]uv run python scripts/record_test_audio.py[/green]"
        )
        raise SystemExit(1)

    providers = providers_from_arg(args.providers)
    console.print(
        f"[bold]Loading {args.model} ({config.params_m}M) ONNX ...[/bold]\n"
        f"providers={providers}"
    )
    asr = VietnameseASR(model_key=args.model, providers=providers)
    console.print("[green]Loaded[/green]")

    _ = asr.transcribe_file(test_files[0])

    table = Table(title=f"ASR Smoke Test Results — {args.model}")
    table.add_column("File", style="cyan")
    table.add_column("Audio (ms)", justify="right")
    table.add_column("Latency (ms)", justify="right", style="yellow")
    table.add_column("RTF", justify="right", style="magenta")
    table.add_column("Avg logprob", justify="right")
    table.add_column("Transcript", style="white")

    for path in test_files:
        result = asr.transcribe_file(path)
        table.add_row(
            path.name,
            f"{result.audio_duration_ms:.0f}",
            f"{result.latency_ms:.0f}",
            f"{result.rtf:.2f}",
            f"{result.avg_logprob:.3f}",
            result.text,
        )

    console.print(table)


if __name__ == "__main__":
    main()
