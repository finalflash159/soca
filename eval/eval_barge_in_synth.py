"""Run synthesized VN barge-in with real-RIR echo (P3.1 Pha C, Tier 1 synth).

Complements ``eval_conversation`` (real AEC-Challenge echo, no onset) by measuring
what real recordings cannot: **stop-latency** (fire − known onset) and the
**backchannel** false-fire probe. Uses the same production AEC + Silero, so the two
Tier 1 runs are directly comparable.

    uv run python -m eval.eval_barge_in_synth --n 80 --onset-ms 1000 --alpha 0.5

Deterministic: replay time is the frame index; scenarios are seeded.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import click

from eval.barge_in_replay import BargeInDecider, EchoCanceller, SpeechProb
from eval.conversation_metrics import (
    SynthBargeOutcome,
    SynthBargeReport,
    synth_barge_report,
)
from eval.eval_conversation import _make_real_adapters
from eval.scenarios_barge_in_synth import SynthScenario, build_scenarios
from local import config as cfg


def evaluate_synth(
    scenarios: list[SynthScenario],
    *,
    aec_factory: Callable[[], EchoCanceller],
    vad: SpeechProb,
    vad_reset: Callable[[], None],
    sustained_ms: float,
    vad_threshold: float,
) -> list[SynthBargeOutcome]:
    """Replay each synth scenario through a fresh AEC + reset VAD; one outcome each."""
    outcomes: list[SynthBargeOutcome] = []
    for scenario in scenarios:
        vad_reset()
        decider = BargeInDecider(
            aec=aec_factory(),
            vad=vad,
            sustained_ms=sustained_ms,
            vad_threshold=vad_threshold,
        )
        result = decider.run(scenario.far, scenario.near)
        outcomes.append(
            SynthBargeOutcome(
                kind=scenario.kind,
                expected_interrupt=scenario.expected_interrupt,
                interrupted=result.interrupted,
                onset_ms=scenario.onset_ms,
                interrupt_ms=result.interrupt_ms,
            )
        )
    return outcomes


def _report_to_dict(report: SynthBargeReport, meta: dict) -> dict:
    return {
        "metadata": meta,
        "n_echo_only": report.n_echo_only,
        "n_barge_in": report.n_barge_in,
        "n_backchannel": report.n_backchannel,
        "false_interrupt_rate": report.false_interrupt_rate,
        "detection_rate": report.detection_rate,
        "backchannel_fire_rate": report.backchannel_fire_rate,
        "median_latency_ms": report.median_latency_ms,
        "p90_latency_ms": report.p90_latency_ms,
    }


def _print_report(report: SynthBargeReport) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="P3.1 Tier 1 synth - VN barge-in over real-RIR echo")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("echo_only / barge_in / backchannel", f"{report.n_echo_only}/{report.n_barge_in}/{report.n_backchannel}")
    table.add_row("[red]false-interrupt rate[/red]", f"{report.false_interrupt_rate * 100:.1f}%")
    table.add_row("[green]detection rate[/green]", f"{report.detection_rate * 100:.1f}%")
    table.add_row("[yellow]backchannel fire rate[/yellow]", f"{report.backchannel_fire_rate * 100:.1f}%")
    if report.median_latency_ms is not None:
        table.add_row("median stop-latency ms", f"{report.median_latency_ms:.0f}")
        table.add_row("p90 stop-latency ms", f"{report.p90_latency_ms:.0f}")
    console.print(table)


@click.command()
@click.option("--n", default=80, type=int, help="Utterances (× echo_only+barge_in+backchannel).")
@click.option("--onset-ms", default=1000.0, type=float, help="User barge-in onset time.")
@click.option("--alpha", default=0.5, type=float, help="Echo level (far*RIR scale).")
@click.option("--backchannel-ms", default=400.0, type=float, help="Backchannel length.")
@click.option("--sustained-ms", default=400.0, type=float, help="Barge-in sustained gate.")
@click.option("--vad-threshold", default=0.7, type=float, help="Silero speech threshold.")
@click.option("--stream-delay-ms", default=40, type=int, help="AEC3 stream delay hint.")
@click.option("--seed", default=cfg.SEED, type=int, help="Deterministic seed.")
def main(
    n: int,
    onset_ms: float,
    alpha: float,
    backchannel_ms: float,
    sustained_ms: float,
    vad_threshold: float,
    stream_delay_ms: int,
    seed: int,
) -> None:
    from rich.console import Console

    console = Console()
    scenarios = build_scenarios(
        n, onset_ms=onset_ms, alpha=alpha, backchannel_ms=backchannel_ms, seed=seed
    )
    console.print(
        f"[bold]Built {len(scenarios)} scenarios "
        f"({n}× echo/barge/backchannel, onset {onset_ms:.0f}ms, alpha {alpha}, seed {seed})[/bold]"
    )

    aec_factory, vad, vad_reset = _make_real_adapters(stream_delay_ms)
    outcomes = evaluate_synth(
        scenarios,
        aec_factory=aec_factory,
        vad=vad,
        vad_reset=vad_reset,
        sustained_ms=sustained_ms,
        vad_threshold=vad_threshold,
    )
    report = synth_barge_report(outcomes)
    _print_report(report)

    meta = {
        "tier": "1_synth",
        "n": n,
        "onset_ms": onset_ms,
        "alpha": alpha,
        "backchannel_ms": backchannel_ms,
        "sustained_ms": sustained_ms,
        "vad_threshold": vad_threshold,
        "seed": seed,
    }
    cfg.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = cfg.EVAL_RESULTS_DIR / "conversation_tier1_synth.json"
    out_path.write_text(
        json.dumps(_report_to_dict(report, meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    console.print(f"[green]✓ Saved {out_path}[/green]")


if __name__ == "__main__":
    main()
