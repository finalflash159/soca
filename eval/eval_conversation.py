"""Run barge-in replay over AEC-Challenge real pairs (P3.1 Pha C, Tier 1).

Wires the *production* echo canceller (WebRTC AEC3) and Silero VAD — the exact
components ``DuplexAecSink`` uses — into ``BargeInDecider`` and replays them,
frame-by-frame, over real device echo. Because replay time is the frame index, the
run is deterministic despite AEC3 being stateful: same pairs, same numbers.

    uv run python -m eval.eval_conversation --n-per-condition 150

Heavy deps (``pywebrtc_audio``, ``silero_vad``, ``torch``) are imported lazily inside
the real adapters, so the module and its aggregation path import cleanly in CI.

Per-turn faithfulness: ``DuplexAecSink`` opens a fresh stream and resets VAD state
each turn, and AEC3 converges from scratch. We mirror that by building a fresh
``AudioProcessor`` and resetting Silero for every pair.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path

import click

from eval.aec_challenge import (
    AecScenario,
    discover_pairs,
    load_pair,
    sample_by_condition,
)
from eval.barge_in_replay import BargeInDecider, EchoCanceller, SpeechProb
from eval.conversation_metrics import BargeInOutcome, BargeInReport, barge_in_report
from local import config as cfg

_SAMPLE_RATE = 16000


def evaluate_scenarios(
    scenarios: Iterable[AecScenario],
    *,
    aec_factory: Callable[[], EchoCanceller],
    vad: SpeechProb,
    vad_reset: Callable[[], None],
    sustained_ms: float,
    vad_threshold: float,
) -> list[BargeInOutcome]:
    """Replay each pair through a fresh AEC + reset VAD; collect one outcome each.

    ``aec_factory`` returns a fresh canceller per pair (AEC3 is stateful — one turn,
    one converge). ``vad_reset`` clears the VAD's per-turn state. Both seams are
    injected so this loop is unit-testable without WebRTC/Silero."""
    outcomes: list[BargeInOutcome] = []
    for scenario in scenarios:
        far, near = load_pair(scenario)
        vad_reset()
        decider = BargeInDecider(
            aec=aec_factory(),
            vad=vad,
            sustained_ms=sustained_ms,
            vad_threshold=vad_threshold,
        )
        result = decider.run(far, near)
        outcomes.append(
            BargeInOutcome(
                condition=scenario.condition,
                expected_interrupt=scenario.expected_interrupt,
                interrupted=result.interrupted,
                interrupt_ms=result.interrupt_ms,
                with_movement=scenario.with_movement,
            )
        )
    return outcomes


def _report_to_dict(report: BargeInReport, meta: dict) -> dict:
    return {
        "metadata": meta,
        "n_total": report.n_total,
        "n_echo_only": report.n_echo_only,
        "n_double_talk": report.n_double_talk,
        "false_interrupt_count": report.false_interrupt_count,
        "detection_count": report.detection_count,
        "false_interrupt_rate": report.false_interrupt_rate,
        "detection_rate": report.detection_rate,
        "missed_rate": report.missed_rate,
        "median_fire_ms": report.median_fire_ms,
        "by_movement": report.by_movement,
    }


def _make_real_adapters(
    stream_delay_ms: int,
) -> tuple[Callable[[], EchoCanceller], SpeechProb, Callable[[], None]]:
    """Build the production WebRTC-AEC factory + Silero VAD callable + its reset."""
    import numpy as np
    import torch
    from pywebrtc_audio import AudioProcessor
    from silero_vad import load_silero_vad

    def aec_factory() -> EchoCanceller:
        processor = AudioProcessor(
            sample_rate=_SAMPLE_RATE,
            num_channels=1,
            echo_cancellation=True,
            noise_suppression=True,
            auto_gain_control=False,
            stream_delay_ms=stream_delay_ms,
        )

        class _Aec:
            def process(self, near: np.ndarray, far: np.ndarray) -> np.ndarray:
                return np.asarray(processor.process(near, far), dtype=np.float32)

        return _Aec()

    model = load_silero_vad()

    def vad(frame: np.ndarray, sample_rate: int) -> float:
        return float(model(torch.from_numpy(frame), sample_rate).item())

    def vad_reset() -> None:
        model.reset_states()

    return aec_factory, vad, vad_reset


def _print_report(report: BargeInReport) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="P3.1 Tier 1 - barge-in on AEC-Challenge real echo")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("echo_only pairs", str(report.n_echo_only))
    table.add_row("double_talk pairs", str(report.n_double_talk))
    table.add_row("[red]false-interrupt rate[/red]", f"{report.false_interrupt_rate * 100:.1f}%")
    table.add_row("[green]detection rate[/green]", f"{report.detection_rate * 100:.1f}%")
    table.add_row("missed rate", f"{report.missed_rate * 100:.1f}%")
    if report.median_fire_ms is not None:
        table.add_row("median fire ms (clip-rel.)", f"{report.median_fire_ms:.0f}")
    console.print(table)
    for label, m in report.by_movement.items():
        console.print(
            f"  {label}: false-int={m['false_interrupt_rate'] * 100:.1f}% "
            f"detect={m['detection_rate'] * 100:.1f}% "
            f"(echo={m['n_echo_only']}, dbl={m['n_double_talk']})"
        )


@click.command()
@click.option("--data-dir", default="data/aec/real", help="AEC-Challenge real/ directory.")
@click.option("--n-per-condition", default=150, type=int, help="Pairs per condition (balanced).")
@click.option("--seed", default=cfg.SEED, type=int, help="Deterministic sampling seed.")
@click.option("--sustained-ms", default=400.0, type=float, help="Barge-in sustained-speech gate.")
@click.option("--vad-threshold", default=0.7, type=float, help="Silero speech threshold.")
@click.option("--stream-delay-ms", default=40, type=int, help="AEC3 stream delay hint.")
def main(
    data_dir: str,
    n_per_condition: int,
    seed: int,
    sustained_ms: float,
    vad_threshold: float,
    stream_delay_ms: int,
) -> None:
    from rich.console import Console

    console = Console()
    root = Path(data_dir)
    if not root.exists():
        raise click.ClickException(f"AEC-Challenge real dir not found: {root}")

    pairs = discover_pairs(root)
    scenarios = sample_by_condition(pairs, n_per_condition=n_per_condition, seed=seed)
    console.print(
        f"[bold]Discovered {len(pairs)} pairs; sampled {len(scenarios)} "
        f"({n_per_condition}/condition, seed {seed})[/bold]"
    )

    aec_factory, vad, vad_reset = _make_real_adapters(stream_delay_ms)
    outcomes = evaluate_scenarios(
        scenarios,
        aec_factory=aec_factory,
        vad=vad,
        vad_reset=vad_reset,
        sustained_ms=sustained_ms,
        vad_threshold=vad_threshold,
    )
    report = barge_in_report(outcomes)
    _print_report(report)

    meta = {
        "tier": 1,
        "data_dir": str(root),
        "n_per_condition": n_per_condition,
        "seed": seed,
        "sustained_ms": sustained_ms,
        "vad_threshold": vad_threshold,
        "stream_delay_ms": stream_delay_ms,
    }
    cfg.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = cfg.EVAL_RESULTS_DIR / "conversation_tier1.json"
    out_path.write_text(
        json.dumps(_report_to_dict(report, meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    console.print(f"[green]✓ Saved {out_path}[/green]")


if __name__ == "__main__":
    main()
