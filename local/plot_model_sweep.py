"""Render cross-model charts from historical raw and experimental BoH runs."""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402

from local import config as cfg  # noqa: E402

# (display name, params in millions, candidate result-JSON filenames relative to
# results dir - first existing wins). Ordered by model size - the sweep x-axis.
# tiny falls back to table7_replication.json, which is what eval_table7 writes
# for the default model when no focused run has been copied aside.
DEFAULT_SWEEP = [
    ("tiny", 39, ("table7_phowhisper_tiny_focused.json", "table7_replication.json")),
    ("base", 74, ("table7_phowhisper_base.json",)),
    ("small", 244, ("table7_phowhisper_small.json",)),
    ("medium", 769, ("table7_phowhisper_medium.json",)),
    ("large", 1550, ("table7_phowhisper_large.json",)),
]

WER_COLOR = "#e6a817"
RTF_COLOR = "#5b8def"
RAW_COLOR = "#d1495b"
EXPERIMENTAL_COLOR = "#2a9d8f"
CURRENT_ABLATION_PAIR = ("production_no_boh", "production_with_boh")
HISTORICAL_ABLATION_PAIR = ("raw", "vad_deloop_boh")


def _rtf(reference_config: dict) -> float:
    """Real-time factor = mean processing time / mean clip duration."""
    durs = [
        diagnostic["speech_duration_ms"]
        for diagnostic in reference_config["diagnostics"]
        if diagnostic.get("speech_duration_ms")
    ]
    if not durs:
        return float("nan")
    return (reference_config["latency_mean_ms"] / 1000) / (st.mean(durs) / 1000)


def _resolve(results_dir: Path, filenames: str | tuple[str, ...]) -> Path | None:
    """First existing candidate file (accepts a single name or a fallback tuple)."""
    candidates = (filenames,) if isinstance(filenames, str) else filenames
    for name in candidates:
        path = results_dir / name
        if path.exists():
            return path
    return None


def _ablation_pair(results: dict) -> tuple[dict, dict, str]:
    if all(code in results for code in CURRENT_ABLATION_PAIR):
        return (
            results[CURRENT_ABLATION_PAIR[0]],
            results[CURRENT_ABLATION_PAIR[1]],
            "production_paired",
        )
    if all(code in results for code in HISTORICAL_ABLATION_PAIR):
        return (
            results[HISTORICAL_ABLATION_PAIR[0]],
            results[HISTORICAL_ABLATION_PAIR[1]],
            "historical_raw_experimental",
        )
    raise ValueError(
        "ASR sweep report must contain one complete ablation pair: "
        f"{CURRENT_ABLATION_PAIR!r} or {HISTORICAL_ABLATION_PAIR!r}; "
        f"found {tuple(sorted(results))!r}"
    )


def load_sweep(results_dir: Path, sweep: list[tuple[str, int, str | tuple[str, ...]]]) -> list[dict]:
    """Read each model's focused JSON into a flat row of the metrics we plot."""
    rows: list[dict] = []
    for name, params_m, filenames in sweep:
        path = _resolve(results_dir, filenames)
        if path is None:
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        reference, experimental, schema = _ablation_pair(report["results"])
        rows.append(
            {
                "name": name,
                "params_m": params_m,
                "ablation_schema": schema,
                "wer_reference": reference["wer"] * 100,
                "cer_reference": reference["cer"] * 100,
                "halluc_reference": reference["hallucination_rate"] * 100,
                "halluc_experimental": experimental["hallucination_rate"] * 100,
                "rtf": _rtf(reference),
            }
        )
    return rows


def _labels(rows: list[dict]) -> list[str]:
    return [f"{r['name']}\n{r['params_m']}M" for r in rows]


def plot_wer_rtf(rows: list[dict], out: Path) -> None:
    """WER(raw) falling and RTF rising as size grows, with the real-time line."""
    x = range(len(rows))
    wer = [r["wer_reference"] for r in rows]
    rtf = [r["rtf"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(list(x), wer, "o-", color=WER_COLOR, linewidth=2, label="WER (reference) %")
    ax1.set_ylabel("WER (reference) %", color=WER_COLOR)
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
    ax1.set_title("Accuracy saturates, compute explodes - PhoWhisper size sweep")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_halluc(rows: list[dict], out: Path) -> None:
    """Compare raw output with the historical experimental BoH configuration."""
    x = range(len(rows))
    reference = [r["halluc_reference"] for r in rows]
    experimental = [r["halluc_experimental"] for r in rows]
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        [i - width / 2 for i in x],
        reference,
        width,
        label="reference",
        color=RAW_COLOR,
    )
    ax.bar(
        [i + width / 2 for i in x],
        experimental,
        width,
        label="production + experimental BoH",
        color=EXPERIMENTAL_COLOR,
    )
    for i, value in enumerate(experimental):
        ax.text(
            i + width / 2,
            value + 1.5,
            f"{value:.1f}",
            ha="center",
            fontsize=8,
            color=EXPERIMENTAL_COLOR,
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(_labels(rows))
    ax.set_ylabel("Hallucination %")
    ax.set_ylim(0, 108)
    ax.set_title("Historical hallucination ablation by PhoWhisper size")
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
            "Run local.eval_table7 --model <size> "
            "--configs production_no_boh,production_with_boh first."
        )
    written = render_sweep(rows, Path(outdir))
    for path in written:
        click.echo(f"✓ {path}")


if __name__ == "__main__":
    main()
