"""Compare endpoint policies on synthesized VN turn-taking (P3.1 Pha C, Tier 2).

Drives FLEURS-derived scenarios (clean ends + within-turn pauses) through
``TurnEndpointDecider`` with the *production* Silero VAD and Smart-Turn v3.2, and
scores each policy on the same items:

    fixed    constant endpoint_silence_ms (non-adaptive baseline)
    p_based  floor + span·P(still-speaking), the adaptive path

Replay time is the frame index, so the comparison is deterministic. The point is the
trade-off, not a single number: a patient policy avoids cutting into pauses but
over-waits on clean ends.

    uv run python -m eval.eval_turn_taking --n-utterances 60 --pause-ms 800

Heavy deps (silero_vad, Smart-Turn ONNX, torch) load lazily in the adapters, so the
module imports cleanly in CI.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import click

from eval.barge_in_replay import (
    POLICY_FIXED,
    POLICY_P_BASED,
    SpeechProb,
    TurnDetector,
    TurnEndpointDecider,
)
from eval.conversation_metrics import TurnOutcome, turn_taking_report
from eval.provenance import config_snapshot, run_provenance
from eval.scenarios_turn_taking import TurnScenario, build_scenarios
from local import config as cfg
from soca.core.endpoint import EndpointConfig

_SMART_TURN_DIR = "models/smart-turn-v3-onnx"


def evaluate_policies(
    scenarios: list[TurnScenario],
    *,
    vad: SpeechProb,
    vad_reset: Callable[[], None],
    turn_detector: TurnDetector,
    policies: tuple[str, ...] = (POLICY_FIXED, POLICY_P_BASED),
) -> list[TurnOutcome]:
    """Run every scenario under every policy; reset the VAD before each replay."""
    outcomes: list[TurnOutcome] = []
    for policy in policies:
        for scenario in scenarios:
            vad_reset()
            decider = TurnEndpointDecider(
                vad=vad,
                policy=policy,
                turn_detector=turn_detector if policy == POLICY_P_BASED else None,
            )
            result = decider.run(scenario.near)
            outcomes.append(
                TurnOutcome(
                    scenario_type=scenario.scenario_type,
                    policy=policy,
                    stopped=result.stopped,
                    stop_ms=result.stop_ms,
                    true_end_ms=scenario.true_end_ms,
                    pause_end_ms=scenario.pause_end_ms,
                )
            )
    return outcomes


def _make_real_adapters() -> tuple[SpeechProb, Callable[[], None], TurnDetector]:
    """Production Silero VAD callable + its reset + Smart-Turn v3.2 detector."""
    import torch
    from silero_vad import load_silero_vad

    from soca.core.smart_turn import SmartTurnDetector

    model = load_silero_vad()

    def vad(frame, sample_rate: int) -> float:
        return float(model(torch.from_numpy(frame), sample_rate).item())

    def vad_reset() -> None:
        model.reset_states()

    turn_detector = SmartTurnDetector(Path(_SMART_TURN_DIR))
    turn_detector.warmup()
    return vad, vad_reset, turn_detector


def _print_report(report: dict) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="P3.1 Tier 2 - endpoint policy on VN turn-taking")
    table.add_column("Policy", style="cyan")
    table.add_column("cut-in rate", justify="right", style="red")
    table.add_column("premature close", justify="right", style="red")
    table.add_column("median over-wait ms", justify="right", style="yellow")
    table.add_column("n (clean/pause)", justify="right")
    for policy, m in report.items():
        over = m["median_over_wait_ms"]
        table.add_row(
            policy,
            f"{m['cut_in_rate'] * 100:.1f}%",
            f"{m['premature_close_rate'] * 100:.1f}%",
            "n/a" if over is None else f"{over:.0f}",
            f"{m['n_clean']}/{m['n_mid_pause']}",
        )
    console.print(table)


@click.command()
@click.option("--n-utterances", default=60, type=int, help="FLEURS utterances (× clean+pause).")
@click.option("--pause-ms", default=800.0, type=float, help="Within-turn pause length.")
@click.option("--seed", default=cfg.SEED, type=int, help="Deterministic sampling seed.")
def main(n_utterances: int, pause_ms: float, seed: int) -> None:
    from rich.console import Console

    console = Console()
    scenarios = build_scenarios(n_utterances, pause_ms=pause_ms, seed=seed)
    console.print(
        f"[bold]Built {len(scenarios)} scenarios "
        f"({n_utterances} utterances × clean+mid_pause@{pause_ms:.0f}ms, seed {seed})[/bold]"
    )

    vad, vad_reset, turn_detector = _make_real_adapters()
    outcomes = evaluate_policies(
        scenarios, vad=vad, vad_reset=vad_reset, turn_detector=turn_detector
    )
    report = turn_taking_report(outcomes)
    _print_report(report)

    # The endpoint constants decide this result as much as the audio does, so they
    # are stamped into the artifact: a later tuning commit then shows up as a diff
    # here instead of silently invalidating whatever the doc quotes.
    meta = run_provenance(
        tier=2,
        n_utterances=n_utterances,
        pause_ms=pause_ms,
        seed=seed,
        endpoint_config=config_snapshot(
            EndpointConfig(),
            ("endpoint_silence_ms", "floor_silence_ms", "ceil_silence_ms"),
        ),
    )
    cfg.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = cfg.EVAL_RESULTS_DIR / "conversation_tier2.json"
    out_path.write_text(
        json.dumps({"metadata": meta, "policies": report}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    console.print(f"[green]✓ Saved {out_path}[/green]")


if __name__ == "__main__":
    main()
