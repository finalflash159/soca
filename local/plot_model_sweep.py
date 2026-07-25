"""Cross-model size-sweep charts for the Table VII replication (P1.1 §6).

Unlike ``plot_table7`` (one model, six configs), this reads the focused
``raw`` + ``vad_deloop_boh`` runs of several PhoWhisper sizes and plots how
accuracy, compute, and hallucination move with model size.

    uv run python -m local.plot_model_sweep

Produces into eval/results/figs/:
  - model_sweep_wer_rtf.png    WER(raw) vs RTF across sizes (accuracy saturates,
                               compute explodes; real-time line at RTF = 1)
  - model_sweep_halluc.png     hallucination raw vs full per size (size never
                               fixes raw hallucination; the pipeline always does)
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402

from local import config as cfg  # noqa: E402

# (display name, params in millions, result-JSON path relative to results dir).
# Ordered by model size — the x-axis of every sweep chart.
DEFAULT_SWEEP = [
    ("tiny", 39, "table7_phowhisper_tiny_focused.json"),
    ("base", 74, "table7_phowhisper_base.json"),
    ("small", 244, "table7_phowhisper_small.json"),
    ("medium", 769, "table7_phowhisper_medium.json"),
    ("large", 1550, "table7_phowhisper_large.json"),
]

WER_COLOR = "#e6a817"
RTF_COLOR = "#5b8def"
RAW_COLOR = "#d1495b"
FULL_COLOR = "#2a9d8f"


def _rtf(raw_config: dict) -> float:
    """Real-time factor = mean processing time / mean clip duration."""
    durs = [d["speech_duration_ms"] for d in raw_config["diagnostics"] if d.get("speech_duration_ms")]
    if not durs:
        return float("nan")
    return (raw_config["latency_mean_ms"] / 1000) / (st.mean(durs) / 1000)


def load_sweep(results_dir: Path, sweep: list[tuple[str, int, str]]) -> list[dict]:
    """Read each model's focused JSON into a flat row of the metrics we plot."""
    rows: list[dict] = []
    for name, params_m, filename in sweep:
        path = results_dir / filename
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        raw = report["results"]["raw"]
        full = report["results"]["vad_deloop_boh"]
        rows.append(
            {
                "name": name,
                "params_m": params_m,
                "wer_raw": raw["wer"] * 100,
                "cer_raw": raw["cer"] * 100,
                "halluc_raw": raw["hallucination_rate"] * 100,
                "halluc_full": full["hallucination_rate"] * 100,
                "rtf": _rtf(raw),
            }
        )
    return rows


def _labels(rows: list[dict]) -> list[str]:
    return [f"{r['name']}\n{r['params_m']}M" for r in rows]


def plot_wer_rtf(rows: list[dict], out: Path) -> None:
    """WER(raw) falling and RTF rising as size grows, with the real-time line."""
    x = range(len(rows))
    wer = [r["wer_raw"] for r in rows]
    rtf = [r["rtf"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(list(x), wer, "o-", color=WER_COLOR, linewidth=2, label="WER (raw) %")
    ax1.set_ylabel("WER (raw) %", color=WER_COLOR)
    ax1.tick_params(axis="y", labelcolor=WER_COLOR)
    ax1.set_ylim(0, max(wer) * 1.25)
    for i, v in enumerate(wer):
        ax1.text(i, v + max(wer) * 0.03, f"{v:.1f}", ha="center", color=WER_COLOR, fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(list(x), rtf, "s-", color=RTF_COLOR, linewidth=2, label="RTF")
    ax2.axhline(1.0, ls="--", color="#888", linewidth=1)
    ax2.text(len(rows) - 1, 1.03, "real-time (RTF=1)", ha="right", color="#666", fontsize=8)
    ax2.set_ylabel("RTF (× real-time)", color=RTF_COLOR)
    ax2.tick_params(axis="y", labelcolor=RTF_COLOR)
    ax2.set_ylim(0, max(rtf) * 1.2)
    for i, v in enumerate(rtf):
        ax2.text(i, v + max(rtf) * 0.03, f"{v:.2f}", ha="center", color=RTF_COLOR, fontsize=8)

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(_labels(rows))
    ax1.set_title("Accuracy saturates, compute explodes — PhoWhisper size sweep")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_halluc(rows: list[dict], out: Path) -> None:
    """Raw vs full hallucination per size: flat 100% raw, pipeline closes it."""
    x = range(len(rows))
    raw = [r["halluc_raw"] for r in rows]
    full = [r["halluc_full"] for r in rows]
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - width / 2 for i in x], raw, width, label="raw (no pipeline)", color=RAW_COLOR)
    ax.bar([i + width / 2 for i in x], full, width, label="full (RobustASR)", color=FULL_COLOR)
    for i, v in enumerate(full):
        ax.text(i + width / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=8, color=FULL_COLOR)
    ax.set_xticks(list(x))
    ax.set_xticklabels(_labels(rows))
    ax.set_ylabel("Hallucination %")
    ax.set_ylim(0, 108)
    ax.set_title("Hallucination: size never fixes it, the pipeline always does")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def render_sweep(rows: list[dict], outdir: Path) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    p1 = outdir / "model_sweep_wer_rtf.png"
    plot_wer_rtf(rows, p1)
    written.append(p1)
    p2 = outdir / "model_sweep_halluc.png"
    plot_halluc(rows, p2)
    written.append(p2)
    return written


@click.command()
@click.option(
    "--results-dir", default=str(cfg.EVAL_RESULTS_DIR),
    help="Directory holding the per-model table7_*.json files.",
)
@click.option(
    "--outdir", default=str(cfg.EVAL_RESULTS_DIR / "figs"),
    help="Directory for the PNG charts.",
)
def main(results_dir: str, outdir: str) -> None:
    rows = load_sweep(Path(results_dir), DEFAULT_SWEEP)
    if len(rows) < 2:
        raise click.ClickException(
            f"Need >=2 model result files in {results_dir} (found {len(rows)}). "
            "Run local.eval_table7 --model <size> --configs raw,vad_deloop_boh first."
        )
    written = render_sweep(rows, Path(outdir))
    for path in written:
        click.echo(f"✓ {path}")


if __name__ == "__main__":
    main()
