"""ASR + LLM text-only pipeline demo.

Records your voice, transcribes, gets a local LLM response.
"""
from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import sounddevice as sd
import soundfile as sf
from rich.console import Console
from rich.panel import Panel

from soca.app.text_runtime import default_text_llm_model_key
from soca.asr import VietnameseASR
from soca.llm import LocalLlamaCppLLM
from soca.llm.registry import LLM_MODEL_REGISTRY

console = Console()
SAMPLE_RATE = 16000
RECORD_DURATION = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record speech and run ASR -> LLM text demo.")
    parser.add_argument(
        "--llm-model",
        default=default_text_llm_model_key(),
        choices=sorted(LLM_MODEL_REGISTRY),
        help="LLM registry key to load.",
    )
    parser.add_argument("--record-seconds", type=float, default=RECORD_DURATION)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    console.print("[bold]Loading models...[/bold]")
    asr = VietnameseASR(num_threads=4)
    llm = LocalLlamaCppLLM(model_key=args.llm_model, n_threads=8, n_gpu_layers=-1)
    console.print(f"[green]✓ Models loaded[/green] LLM={args.llm_model}\n")

    while True:
        input(f"\n>>> Press ENTER to record ({args.record_seconds:g} seconds), or Ctrl+C to quit...")
        console.print(f"[yellow]Recording {args.record_seconds:g}s...[/yellow]")
        audio = sd.rec(
            int(args.record_seconds * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        audio = audio.flatten()

        tmp_path = Path("/tmp/soca_last.wav")
        sf.write(str(tmp_path), audio, SAMPLE_RATE)
        console.print(f"[dim]Saved to {tmp_path}[/dim]")

        t0 = time.perf_counter()
        asr_result = asr.transcribe(audio)
        t_asr = (time.perf_counter() - t0) * 1000

        console.print(
            Panel(
                f"[cyan]{asr_result.text}[/cyan]\n\n"
                f"[dim]ASR latency: {t_asr:.0f} ms[/dim]",
                title="Bạn nói",
                border_style="cyan",
            )
        )

        if not asr_result.text.strip():
            console.print("[red]Empty transcript; speak louder?[/red]")
            continue

        t0 = time.perf_counter()
        llm_result = llm.generate(asr_result.text, max_tokens=120, temperature=0.7)
        t_llm = (time.perf_counter() - t0) * 1000

        console.print(
            Panel(
                f"[yellow]{llm_result.text}[/yellow]\n\n"
                f"[dim]LLM TTFT: {llm_result.ttft_ms:.0f} ms | "
                f"Total: {t_llm:.0f} ms | {llm_result.tokens_per_second:.1f} tok/s[/dim]",
                title="SoCa trả lời",
                border_style="yellow",
            )
        )

        total = t_asr + t_llm
        console.print(f"\n[bold]End-to-end (text): {total:.0f} ms[/bold]")


if __name__ == "__main__":
    raise SystemExit(main())
