"""CLI: calibrate hallucination heuristic thresholds from real Vietnamese speech.

= notebooks/03_threshold_calibration.ipynb (Mac CLI version).

Computes the empirical distribution of:
  - unigram repetition ratio
  - 3-gram repetition ratio
  - chars-per-100ms density

on real FLEURS vi_vn ground-truth transcripts (no ASR involved). Picks
thresholds at p99 + safety margin so the heuristic rejects ~0% of real
speech as "hallucination".

Why ground truth and not ASR output? We want to characterize what NATURAL
Vietnamese speech looks like by these metrics. ASR adds noise to the
measurement; ground truth is the source-of-truth ceiling.

Output: data/asr/threshold_calibration.json

Usage:
    uv run python -m local.download_fleurs --target 200   # one-time
    uv run python -m local.calibrate_thresholds
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import click
import numpy as np
import soundfile as sf
from rich.console import Console
from rich.table import Table

from local import config as cfg
from soca.asr.hallucination_heuristics import n_gram_repetition, repetition_ratio

console = Console()


def percentile_summary(values: list[float], name: str) -> dict:
    arr = np.array(values)
    return {
        "metric": name,
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


@click.command()
@click.option(
    "--repetition-margin", default=0.15, type=float,
    help="Safety margin above p99 for unigram/n-gram thresholds.",
)
@click.option(
    "--density-margin", default=0.5, type=float,
    help="Safety margin above p99 for chars-per-100ms (looser — varies by speaker).",
)
def main(repetition_margin: float, density_margin: float) -> None:
    if not cfg.FLEURS_MANIFEST.exists():
        raise click.ClickException(
            f"FLEURS manifest not found: {cfg.FLEURS_MANIFEST}\n"
            "Run: uv run python -m local.download_fleurs --target 200"
        )

    rows: list[dict] = []
    with cfg.FLEURS_MANIFEST.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    console.print(f"Calibrating on [bold]{len(rows)}[/bold] FLEURS vi samples")

    repetition_scores: list[float] = []
    ngram_scores: list[float] = []
    chars_per_100ms_scores: list[float] = []

    for row in rows:
        text = row["ground_truth"]
        repetition_scores.append(repetition_ratio(text))
        ngram_scores.append(n_gram_repetition(text))

        # Prefer manifest duration; fall back to reading the file.
        duration_s = row.get("duration_s")
        if duration_s is None:
            info = sf.info(cfg.FLEURS_WAV_DIR / row["filename"])
            duration_s = info.frames / info.samplerate
        duration_ms = duration_s * 1000
        if duration_ms > 0:
            chars_per_100ms_scores.append(len(text) / (duration_ms / 100))

    stats = [
        percentile_summary(repetition_scores, "unigram_repetition"),
        percentile_summary(ngram_scores, "3gram_repetition"),
        percentile_summary(chars_per_100ms_scores, "chars_per_100ms"),
    ]

    recommended = {
        "repetition_thresh": stats[0]["p99"] * (1 + repetition_margin),
        "ngram_repetition_thresh": stats[1]["p99"] * (1 + repetition_margin),
        "chars_per_100ms_max": stats[2]["p99"] * (1 + density_margin),
    }

    table = Table(title=f"Threshold calibration on {len(rows)} FLEURS vi samples")
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", justify="right")
    table.add_column("P50", justify="right")
    table.add_column("P90", justify="right")
    table.add_column("P95", justify="right", style="yellow")
    table.add_column("P99", justify="right", style="yellow")
    table.add_column("Max", justify="right", style="red")
    table.add_column("Recommended", justify="right", style="green")
    for stat in stats:
        key = {
            "unigram_repetition": "repetition_thresh",
            "3gram_repetition": "ngram_repetition_thresh",
            "chars_per_100ms": "chars_per_100ms_max",
        }[stat["metric"]]
        table.add_row(
            stat["metric"],
            f"{stat['mean']:.3f}",
            f"{stat['p50']:.3f}",
            f"{stat['p90']:.3f}",
            f"{stat['p95']:.3f}",
            f"{stat['p99']:.3f}",
            f"{stat['max']:.3f}",
            f"{recommended[key]:.3f}",
        )
    console.print(table)

    payload = {
        "dataset": cfg.FLEURS_REPO,
        "language": cfg.FLEURS_LANG,
        "split": cfg.FLEURS_SPLIT,
        "n_samples": len(rows),
        "stats": stats,
        "recommended_thresholds": recommended,
        "margins": {
            "repetition": repetition_margin,
            "chars_per_100ms": density_margin,
        },
        "created_by": "local.calibrate_thresholds",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "usage_note": (
            "Recommended values are p99 + margin. Copy these into the defaults "
            "of soca.asr.hallucination_heuristics.check_heuristics, or pass "
            "them via kwargs at call site."
        ),
    }
    cfg.ASR_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cfg.THRESHOLD_CALIBRATION_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    console.print(f"\n[green]✓ Saved {cfg.THRESHOLD_CALIBRATION_PATH}[/green]")


if __name__ == "__main__":
    main()
