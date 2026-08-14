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
from dataclasses import replace
from hashlib import sha256
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
_FLOOR_SWEEP_MS = (1000, 1200, 1400, 1600, 1800)
_MAX_CUT_IN_RATE = 0.05
_MAX_PREMATURE_CLOSE_RATE = 0.05


def evaluate_policies(
    scenarios: list[TurnScenario],
    *,
    vad: SpeechProb,
    vad_reset: Callable[[], None],
    turn_detector: TurnDetector,
    policies: tuple[str, ...] = (POLICY_FIXED, POLICY_P_BASED),
    config: EndpointConfig | None = None,
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
                config=config,
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


def evaluate_floor_sweep(
    scenarios: list[TurnScenario],
    *,
    vad: SpeechProb,
    vad_reset: Callable[[], None],
    turn_detector: TurnDetector,
    floors_ms: tuple[int, ...] = _FLOOR_SWEEP_MS,
) -> dict[str, dict[str, float | int | None]]:
    """Run paired p-based replays at every requested floor."""
    report: dict[str, dict[str, float | int | None]] = {}
    for floor_ms in floors_ms:
        config = replace(EndpointConfig(), floor_silence_ms=floor_ms)
        outcomes = evaluate_policies(
            scenarios,
            vad=vad,
            vad_reset=vad_reset,
            turn_detector=turn_detector,
            policies=(POLICY_P_BASED,),
            config=config,
        )
        report[str(floor_ms)] = turn_taking_report(outcomes)[POLICY_P_BASED]
    return report


def choose_floor(
    report: dict[str, dict[str, float | int | None]],
    *,
    max_cut_in_rate: float = _MAX_CUT_IN_RATE,
    max_premature_close_rate: float = _MAX_PREMATURE_CLOSE_RATE,
) -> int | None:
    """Select the lowest-latency passing floor; never fall back implicitly."""
    candidates: list[tuple[float, int]] = []
    for raw_floor, metrics in report.items():
        over_wait = metrics.get("median_over_wait_ms")
        if over_wait is None:
            continue
        if float(metrics["cut_in_rate"]) > max_cut_in_rate:
            continue
        if float(metrics["premature_close_rate"]) > max_premature_close_rate:
            continue
        candidates.append((float(over_wait), int(raw_floor)))
    if not candidates:
        return None
    return min(candidates)[1]


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
@click.option(
    "--floor-ms",
    "floors_ms",
    multiple=True,
    default=_FLOOR_SWEEP_MS,
    type=click.IntRange(min=1),
    show_default=True,
)
def main(n_utterances: int, pause_ms: float, seed: int, floors_ms: tuple[int, ...]) -> None:
    from rich.console import Console

    console = Console()
    scenarios = build_scenarios(n_utterances, pause_ms=pause_ms, seed=seed)
    console.print(
        f"[bold]Built {len(scenarios)} scenarios "
        f"({n_utterances} utterances × clean+mid_pause@{pause_ms:.0f}ms, seed {seed})[/bold]"
    )

    vad, vad_reset, turn_detector = _make_real_adapters()
    fixed_outcomes = evaluate_policies(
        scenarios,
        vad=vad,
        vad_reset=vad_reset,
        turn_detector=turn_detector,
        policies=(POLICY_FIXED,),
    )
    fixed_report = turn_taking_report(fixed_outcomes)[POLICY_FIXED]
    sweep_report = evaluate_floor_sweep(
        scenarios,
        vad=vad,
        vad_reset=vad_reset,
        turn_detector=turn_detector,
        floors_ms=tuple(dict.fromkeys(floors_ms)),
    )
    selected_floor_ms = choose_floor(sweep_report)
    current_floor_ms = EndpointConfig().floor_silence_ms
    _print_report(
        {
            POLICY_FIXED: fixed_report,
            **{f"{POLICY_P_BASED}@{floor}ms": row for floor, row in sweep_report.items()},
        }
    )

    # The endpoint constants decide this result as much as the audio does, so they
    # are stamped into the artifact: a later tuning commit then shows up as a diff
    # here instead of silently invalidating whatever the doc quotes.
    meta = run_provenance(
        tier=2,
        n_utterances=n_utterances,
        pause_ms=pause_ms,
        seed=seed,
        fleurs_manifest_sha256=_file_sha256(cfg.FLEURS_MANIFEST),
        smart_turn_model_sha256=_file_sha256(
            Path(_SMART_TURN_DIR) / "smart-turn-v3.2-cpu.onnx"
        ),
        endpoint_base_config=config_snapshot(
            EndpointConfig(), ("endpoint_silence_ms", "floor_silence_ms", "ceil_silence_ms")
        ),
        floor_sweep_ms=list(dict.fromkeys(floors_ms)),
    )
    cfg.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = cfg.EVAL_RESULTS_DIR / "conversation_tier2_sweep.json"
    out_path.write_text(
        json.dumps(
            {
                "metadata": meta,
                "fixed": fixed_report,
                "p_based_by_floor": sweep_report,
                "decision": {
                    "selected_floor_ms": selected_floor_ms,
                    "current_floor_ms": current_floor_ms,
                    "max_cut_in_rate": _MAX_CUT_IN_RATE,
                    "max_premature_close_rate": _MAX_PREMATURE_CLOSE_RATE,
                    "gate_status": "pass" if selected_floor_ms is not None else "fail",
                    "disposition": (
                        "no_passing_floor"
                        if selected_floor_ms is None
                        else "keep_current"
                        if selected_floor_ms == current_floor_ms
                        else "change_recommended"
                    ),
                    "turn_taking_blocker_closed": (
                        selected_floor_ms is not None and selected_floor_ms < current_floor_ms
                    ),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"[green]✓ Saved {out_path}[/green]")


if __name__ == "__main__":
    main()
