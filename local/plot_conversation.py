"""Charts for the P3.1 conversational eval (Pha D).

Reads the three result JSONs written by the eval drivers and renders one chart per
tier into notes/figs/. Titles use a plain hyphen (project style).

    uv run python -m local.plot_conversation

Inputs (eval/results/, gitignored):
    conversation_tier1.json        real AEC-Challenge barge-in
    conversation_tier2.json        turn-taking policy comparison
    conversation_tier1_synth.json  synth VN barge-in (latency + backchannel)
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from local import config as cfg  # noqa: E402

FALSE_COLOR = "#d1495b"
DETECT_COLOR = "#2a9d8f"
WAIT_COLOR = "#e6a817"
BACK_COLOR = "#8d6cab"


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def plot_tier1_real(report: dict, out: Path) -> None:
    """False-interrupt vs detection on real echo, split static/moving."""
    groups = ["overall", "static", "moving"]
    mv = report.get("by_movement", {})
    false_r = [
        report["false_interrupt_rate"] * 100,
        mv.get("static", {}).get("false_interrupt_rate", 0) * 100,
        mv.get("moving", {}).get("false_interrupt_rate", 0) * 100,
    ]
    detect = [
        report["detection_rate"] * 100,
        mv.get("static", {}).get("detection_rate", 0) * 100,
        mv.get("moving", {}).get("detection_rate", 0) * 100,
    ]
    x = range(len(groups))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], false_r, width, label="false-interrupt %", color=FALSE_COLOR)
    ax.bar([i + width / 2 for i in x], detect, width, label="detection %", color=DETECT_COLOR)
    for i, v in enumerate(false_r):
        ax.text(i - width / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8, color=FALSE_COLOR)
    for i, v in enumerate(detect):
        ax.text(i + width / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8, color=DETECT_COLOR)
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.set_ylabel("%")
    ax.set_ylim(0, 108)
    ax.set_title("Barge-in on real device echo - AEC-Challenge (n=300)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_tier2_policy(report: dict, out: Path) -> None:
    """Cut-in / premature (bars, left) vs over-wait ms (line, right) per policy."""
    policies = list(report.keys())
    cut_in = [report[p]["cut_in_rate"] * 100 for p in policies]
    premature = [report[p]["premature_close_rate"] * 100 for p in policies]
    over_wait = [report[p]["median_over_wait_ms"] or 0 for p in policies]

    x = range(len(policies))
    width = 0.38
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar([i - width / 2 for i in x], cut_in, width, label="cut-in %", color=FALSE_COLOR)
    ax1.bar([i + width / 2 for i in x], premature, width, label="premature-close %", color=WAIT_COLOR)
    ax1.set_ylabel("error rate %")
    ax1.set_ylim(0, 108)
    for i, v in enumerate(cut_in):
        ax1.text(i - width / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8, color=FALSE_COLOR)
    for i, v in enumerate(premature):
        ax1.text(i + width / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8, color=WAIT_COLOR)

    ax2 = ax1.twinx()
    ax2.plot(list(x), over_wait, "o-", color=DETECT_COLOR, linewidth=2, label="median over-wait ms")
    ax2.set_ylabel("median over-wait ms", color=DETECT_COLOR)
    ax2.tick_params(axis="y", labelcolor=DETECT_COLOR)
    ax2.set_ylim(0, max(over_wait) * 1.4 if over_wait else 1)
    for i, v in enumerate(over_wait):
        ax2.text(i, v + max(over_wait) * 0.04, f"{v:.0f}", ha="center", fontsize=8, color=DETECT_COLOR)

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(policies)
    ax1.set_title("Turn-taking policy - eager fixed vs adaptive p_based (VN, n=120)")
    ax1.legend(loc="upper center")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_tier1_synth(report: dict, out: Path) -> None:
    """Synth barge-in: outcome rates + a latency annotation."""
    labels = ["false-interrupt", "detection", "backchannel-fire"]
    vals = [
        report["false_interrupt_rate"] * 100,
        report["detection_rate"] * 100,
        report["backchannel_fire_rate"] * 100,
    ]
    colors = [FALSE_COLOR, DETECT_COLOR, BACK_COLOR]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(list(x), vals, 0.5, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("%")
    ax.set_ylim(0, 108)
    med = report.get("median_latency_ms")
    p90 = report.get("p90_latency_ms")
    subtitle = (
        f"median stop-latency {med:.0f}ms / p90 {p90:.0f}ms" if med is not None else "no latency"
    )
    ax.set_title(f"Synth VN barge-in over real-RIR echo - {subtitle}")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


@click.command()
@click.option("--results-dir", default=str(cfg.EVAL_RESULTS_DIR))
@click.option("--outdir", default="notes/figs")
def main(results_dir: str, outdir: str) -> None:
    rdir = Path(results_dir)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    tier1 = _load(rdir / "conversation_tier1.json")
    if tier1:
        plot_tier1_real(tier1, out / "conversation_tier1_real.png")
        written.append("conversation_tier1_real.png")
    tier2 = _load(rdir / "conversation_tier2.json")
    if tier2:
        plot_tier2_policy(tier2["policies"], out / "conversation_tier2_policy.png")
        written.append("conversation_tier2_policy.png")
    synth = _load(rdir / "conversation_tier1_synth.json")
    if synth:
        plot_tier1_synth(synth, out / "conversation_tier1_synth.png")
        written.append("conversation_tier1_synth.png")

    if not written:
        raise click.ClickException(f"No result JSONs found in {rdir}")
    for name in written:
        click.echo(f"✓ {out / name}")


if __name__ == "__main__":
    main()
