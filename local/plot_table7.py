"""Render Table VII replication charts from a benchmark JSON (P1.1 Pha D).

    uv run python -m local.plot_table7                                  # tiny (canonical)
    uv run python -m local.plot_table7 --input eval/results/table7_phowhisper_large.json

Produces into eval/results/figs/:
  - wer_vs_halluc_<model>.png      recognition cost vs hallucination safety, per config
  - stage_contribution_<model>.png which stage catches non-speech (full pipeline)
  - halluc_by_subtype_<model>.png  pure vs speech-like hallucination, per config
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402

from eval.robustness_metrics import STAGE_ORDER  # noqa: E402
from local import config as cfg  # noqa: E402

CONFIG_SHORT = {
    "raw": "raw",
    "deloop": "+deloop",
    "vad": "+vad",
    "boh": "+boh",
    "deloop_boh": "deloop+boh",
    "vad_deloop_boh": "full",
}


def _short(code: str) -> str:
    return CONFIG_SHORT.get(code, code)


def plot_wer_vs_halluc(results: dict, model: str, out: Path) -> None:
    codes = list(results)
    wer = [results[c]["wer"] * 100 for c in codes]
    halluc = [results[c]["hallucination_rate"] * 100 for c in codes]
    frej = [results[c]["robustness"]["false_reject_rate"] * 100 for c in codes]

    x = range(len(codes))
    width = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([i - width for i in x], wer, width, label="WER %", color="#e6a817")
    ax.bar(list(x), halluc, width, label="Hallucination %", color="#d1495b")
    ax.bar([i + width for i in x], frej, width, label="False-reject %", color="#5b8def")
    ax.set_xticks(list(x))
    ax.set_xticklabels([_short(c) for c in codes], rotation=20, ha="right")
    ax.set_ylabel("%")
    ax.set_title(f"Recognition cost vs hallucination safety - {model}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_stage_contribution(full: dict, model: str, out: Path) -> None:
    breakdown = full["robustness"]["noise_stage_breakdown"]
    n_noise = max(full["robustness"]["n_noise"], 1)
    stages = [s for s in STAGE_ORDER if breakdown.get(s)]
    pct = [breakdown[s] / n_noise * 100 for s in stages]
    colors = ["#d1495b" if s == "accepted" else "#2a9d8f" for s in stages]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    positions = range(len(stages))
    ax.bar(positions, pct, color=colors)
    ax.set_ylabel("% of non-speech items")
    ax.set_title(f"Which stage catches non-speech (full pipeline) - {model}")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(stages, rotation=20, ha="right")
    for i, v in enumerate(pct):
        ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_halluc_by_subtype(results: dict, model: str, out: Path) -> None:
    codes = list(results)
    subtypes = sorted(
        {s for c in codes for s in results[c]["robustness"]["hallucination_rate_by_subtype"]}
    )
    if not subtypes:
        return
    x = range(len(codes))
    width = 0.8 / max(len(subtypes), 1)
    palette = {"pure": "#2a9d8f", "speech_like": "#d1495b", "unknown": "#888888"}

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for j, sub in enumerate(subtypes):
        vals = [
            results[c]["robustness"]["hallucination_rate_by_subtype"].get(sub, 0.0) * 100
            for c in codes
        ]
        ax.bar([i + j * width for i in x], vals, width, label=sub, color=palette.get(sub))
    ax.set_xticks([i + width * (len(subtypes) - 1) / 2 for i in x])
    ax.set_xticklabels([_short(c) for c in codes], rotation=20, ha="right")
    ax.set_ylabel("Hallucination %")
    ax.set_title(f"Hallucination by noise subtype - {model}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def render_all(report: dict, outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    results = report["results"]
    model = report["metadata"].get("asr_runtime_identity", {}).get("model_key", "model")

    written: list[Path] = []
    p1 = outdir / f"wer_vs_halluc_{model}.png"
    plot_wer_vs_halluc(results, model, p1)
    written.append(p1)

    if "vad_deloop_boh" in results:
        p2 = outdir / f"stage_contribution_{model}.png"
        plot_stage_contribution(results["vad_deloop_boh"], model, p2)
        written.append(p2)

    p3 = outdir / f"halluc_by_subtype_{model}.png"
    plot_halluc_by_subtype(results, model, p3)
    if p3.exists():
        written.append(p3)
    return written


@click.command()
@click.option(
    "--input", "input_path",
    default=str(cfg.EVAL_RESULTS_DIR / "table7_replication.json"),
    help="Path to a table7 benchmark JSON.",
)
@click.option(
    "--outdir", default=str(cfg.EVAL_RESULTS_DIR / "figs"),
    help="Directory for the PNG charts.",
)
def main(input_path: str, outdir: str) -> None:
    report = json.loads(Path(input_path).read_text(encoding="utf-8"))
    written = render_all(report, Path(outdir))
    for path in written:
        click.echo(f"✓ {path}")


if __name__ == "__main__":
    main()
