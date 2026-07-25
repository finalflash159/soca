"""CLI: replicate Table VII of Barański et al. ICASSP 2025 for Vietnamese.

= notebooks/04_table7_replication.ipynb (Mac CLI version).

Compares 6 configurations on a mix of real speech + non-speech:

    raw              — ASR only, no post-processing
    deloop           — ASR + consecutive-repeat collapse
    vad              — Silero VAD pre-filter + ASR
    boh              — ASR + Aho-Corasick BoH match
    deloop_boh       — ASR + de-loop + BoH
    vad_deloop_boh   — production RobustASR pipeline
                       (VAD + ASR + confidence guard + de-loop + BoH + heuristics)

Reports per config:
  - WER + CER on real Vietnamese speech (FLEURS vi_vn)
  - hallucination rate on non-speech (any non-empty output is a hallucination)
  - p50/p95 latency per sample

This is the centerpiece deliverable for the D2.5 CV bullet — replicating
the paper's mitigation pattern for Vietnamese PhoWhisper-tiny.

Output: eval/results/table7_replication.json

Usage:
    # one-time setup
    uv run python -m local.collect_noise          # build noise dataset
    uv run python -m local.download_fleurs        # build speech dataset
    uv run python -m local.build_boh              # build BoH artifact

    # main run
    uv run python -m local.eval_table7
    uv run python -m local.eval_table7 --n-speech 30 --n-noise 10  # smoke
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import click
import librosa
import numpy as np
import soundfile as sf
from jiwer import cer, wer
from rich.console import Console
from rich.progress import track
from rich.table import Table

from eval.robustness_metrics import (
    STAGE_ORDER,
    Diagnostic,
    RobustnessReport,
    compute_robustness_report,
)
from local import config as cfg
from soca.asr import (
    RobustASR,
    SpeechDetector,
    VietnameseASR,
    VietnameseBoH,
    remove_consecutive_repeats,
)
from soca.asr.registry import ASR_MODEL_REGISTRY, DEFAULT_ASR_MODEL_KEY

console = Console()


def _to_diagnostic(d: dict) -> Diagnostic:
    return Diagnostic(
        kind=d["kind"],
        rejection_reason=d.get("rejection_reason", ""),
        final_text=d.get("final_text", ""),
        ground_truth=d.get("ground_truth", ""),
        subtype=d.get("subtype", "unknown"),
    )


def _robustness_report(diagnostics: list[dict]) -> RobustnessReport:
    return compute_robustness_report([_to_diagnostic(d) for d in diagnostics])


def _report_to_dict(report: RobustnessReport) -> dict:
    return {
        "n_speech": report.n_speech,
        "n_noise": report.n_noise,
        "false_reject_count": report.false_reject_count,
        "false_reject_rate": report.false_reject_rate,
        "hallucination_count": report.hallucination_count,
        "hallucination_rate": report.hallucination_rate,
        "catch_rate": report.catch_rate,
        "wer_accepted": report.wer,
        "cer_accepted": report.cer,
        "noise_stage_breakdown": report.noise_stage_breakdown,
        "hallucination_rate_by_subtype": report.hallucination_rate_by_subtype,
    }


def _print_stage_breakdown(report: RobustnessReport) -> None:
    """Show which stage caught each non-speech item, for the full pipeline."""
    breakdown = report.noise_stage_breakdown
    if not report.n_noise:
        return

    table = Table(title="Stage contribution — non-speech catches (full pipeline)")
    table.add_column("Stage", style="cyan")
    table.add_column("Caught", justify="right")
    table.add_column("% of noise", justify="right")
    for stage in STAGE_ORDER:
        count = breakdown.get(stage, 0)
        if not count:
            continue
        style = "red" if stage == "accepted" else None
        label = f"{stage} (leaked through)" if stage == "accepted" else stage
        table.add_row(
            f"[red]{label}[/red]" if style else label,
            str(count),
            f"{count / report.n_noise * 100:.1f}%",
        )
    console.print(table)
    console.print(
        f"  catch-rate={report.catch_rate * 100:.1f}%  "
        f"false-reject={report.false_reject_rate * 100:.2f}%  "
        f"WER(accepted)={_fmt_pct(report.wer)}"
    )
    if report.hallucination_rate_by_subtype:
        parts = ", ".join(
            f"{sub}={rate * 100:.1f}%"
            for sub, rate in sorted(report.hallucination_rate_by_subtype.items())
        )
        console.print(f"  hallucination-rate by subtype: {parts}")


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"

CONFIG_CODES = ("raw", "deloop", "vad", "boh", "deloop_boh", "vad_deloop_boh")
CONFIG_LABELS = {
    "raw": "(1) Raw ASR",
    "deloop": "(2) De-loop only",
    "vad": "(3) Silero VAD only",
    "boh": "(4) BoH only",
    "deloop_boh": "(5) De-loop + BoH",
    "vad_deloop_boh": "(6) Full pipeline (RobustASR production path)",
}


@dataclass
class Item:
    audio: np.ndarray
    ground_truth: str
    duration_ms: float
    kind: str  # "speech" | "noise"
    subtype: str = "unknown"  # noise: "pure" | "speech_like"; speech: "unknown"


def load_audio(path: Path) -> np.ndarray:
    """Load + resample to 16kHz mono float32."""
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != cfg.SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=cfg.SAMPLE_RATE)
    return audio


def load_speech_items(n: int) -> list[Item]:
    if not cfg.FLEURS_MANIFEST.exists():
        raise click.ClickException(
            f"FLEURS manifest missing: {cfg.FLEURS_MANIFEST}\n"
            "Run: uv run python -m local.download_fleurs"
        )

    items: list[Item] = []
    with cfg.FLEURS_MANIFEST.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            audio = load_audio(cfg.FLEURS_WAV_DIR / row["filename"])
            items.append(
                Item(
                    audio=audio,
                    ground_truth=row["ground_truth"],
                    duration_ms=len(audio) / cfg.SAMPLE_RATE * 1000,
                    kind="speech",
                )
            )
            if len(items) >= n:
                break
    return items


def load_noise_items(n: int) -> list[Item]:
    if not cfg.NOISE_MANIFEST.exists():
        raise click.ClickException(
            f"Noise manifest missing: {cfg.NOISE_MANIFEST}\n"
            "Run: uv run python -m local.collect_noise"
        )

    rows: list[dict] = []
    with cfg.NOISE_MANIFEST.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    # Deterministic shuffle so any subset stays stratified across pure/speech_like
    # (babble is appended at the tail of the manifest). Full runs load everything.
    random.Random(cfg.SEED).shuffle(rows)

    items: list[Item] = []
    for row in rows[:n]:
        audio = load_audio(cfg.NOISE_ROOT / row["path"])
        items.append(
            Item(
                audio=audio,
                ground_truth="",  # noise should produce nothing
                duration_ms=len(audio) / cfg.SAMPLE_RATE * 1000,
                kind="noise",
                subtype=row.get("subtype", "pure"),
            )
        )
    return items


def run_config(
    code: str,
    items: list[Item],
    asr: VietnameseASR,
    vad: SpeechDetector,
    boh: VietnameseBoH | None,
    robust_asr: RobustASR | None,
) -> dict:
    """Run a single pipeline configuration. Return per-item predictions + metrics."""
    predictions: list[str] = []
    latencies_ms: list[float] = []
    diagnostics: list[dict] = []

    for item in track(items, description=f"  {code}"):
        t0 = time.perf_counter()
        audio = item.audio
        speech_duration_ms = item.duration_ms

        # The final Table VII config must exercise the actual production class.
        # Earlier configs stay manual so they isolate each mitigation stage.
        if code == "vad_deloop_boh":
            if robust_asr is None:
                raise RuntimeError("robust_asr is required for vad_deloop_boh config")
            result = robust_asr.transcribe(audio)
            predictions.append(result.text.strip())
            latencies_ms.append(result.total_latency_ms)
            diagnostics.append(
                {
                    "kind": item.kind,
                    "raw_text": result.raw_text,
                    "final_text": result.text,
                    "rejection_reason": result.rejection_reason,
                    "has_speech": result.has_speech,
                    "was_looping": result.was_looping,
                    "boh_matches": list(result.boh_matches),
                    "avg_logprob": result.avg_logprob,
                    "compression_ratio": result.compression_ratio,
                    "speech_duration_ms": (
                        result.vad.speech_duration_ms if result.vad is not None else 0.0
                    ),
                    "total_latency_ms": result.total_latency_ms,
                }
            )
            continue

        # VAD pre-filter (optional)
        if "vad" in code:
            vad_result = vad.detect(audio)
            if not vad_result.has_speech:
                predictions.append("")
                latencies_ms.append((time.perf_counter() - t0) * 1000)
                diagnostics.append(
                    {
                        "kind": item.kind,
                        "raw_text": "",
                        "final_text": "",
                        "rejection_reason": "no_speech",
                        "has_speech": False,
                        "speech_duration_ms": 0.0,
                    }
                )
                continue
            audio = vad_result.speech_audio
            speech_duration_ms = vad_result.speech_duration_ms

        # ASR
        text = asr.transcribe(audio).text
        raw_text = text

        # De-loop (optional)
        if "deloop" in code:
            text, _ = remove_consecutive_repeats(text)

        # BoH match (optional)
        if "boh" in code and boh is not None:
            text = boh.match_and_clean(text).cleaned_text

        predictions.append(text.strip())
        latencies_ms.append((time.perf_counter() - t0) * 1000)
        diagnostics.append(
            {
                "kind": item.kind,
                "raw_text": raw_text,
                "final_text": text.strip(),
                "rejection_reason": "",
                "has_speech": True,
                "speech_duration_ms": speech_duration_ms,
            }
        )

    # Enrich each diagnostic with its item's ground truth + noise subtype so the
    # JSON is self-sufficient for downstream stage-decomposition (eval.robustness_metrics).
    enriched = [
        {**d, "ground_truth": it.ground_truth, "subtype": getattr(it, "subtype", "unknown")}
        for it, d in zip(items, diagnostics, strict=True)
    ]

    return {
        "predictions": predictions,
        "latencies_ms": latencies_ms,
        "diagnostics": enriched,
    }


def compute_metrics(items: list[Item], predictions: list[str], latencies_ms: list[float]) -> dict:
    speech_refs = [it.ground_truth.lower().strip() for it in items if it.kind == "speech"]
    speech_preds = [
        (p or "<empty>").lower().strip()
        for it, p in zip(items, predictions, strict=False)
        if it.kind == "speech"
    ]
    noise_preds = [p for it, p in zip(items, predictions, strict=False) if it.kind == "noise"]
    n_noise = sum(1 for it in items if it.kind == "noise")

    wer_val = wer(speech_refs, speech_preds) if speech_refs else float("nan")
    cer_val = cer(speech_refs, speech_preds) if speech_refs else float("nan")
    hallucinated = sum(1 for p in noise_preds if p.strip())
    halluc_rate = hallucinated / max(n_noise, 1)

    lat = np.array(latencies_ms)
    return {
        "wer": wer_val,
        "cer": cer_val,
        "hallucination_rate": halluc_rate,
        "n_speech": len(speech_refs),
        "n_noise": n_noise,
        "hallucinated_count": hallucinated,
        "latency_p50_ms": float(np.percentile(lat, 50)),
        "latency_p95_ms": float(np.percentile(lat, 95)),
        "latency_mean_ms": float(lat.mean()),
    }


@click.command()
@click.option("--n-speech", default=50, type=int, help="Number of speech samples (FLEURS vi).")
@click.option("--n-noise", default=20, type=int, help="Number of noise samples.")
@click.option(
    "--configs", default=",".join(CONFIG_CODES),
    help=f"Comma-separated subset of: {','.join(CONFIG_CODES)}",
)
@click.option(
    "--providers", default="auto",
    type=click.Choice(["auto", "cpu"]),
    help="auto = CoreML + CPU fallback (Mac), cpu = force CPU.",
)
@click.option(
    "--model", "model_key", default=DEFAULT_ASR_MODEL_KEY,
    type=click.Choice(sorted(ASR_MODEL_REGISTRY)),
    help="PhoWhisper size to benchmark (robustness x model size).",
)
def main(n_speech: int, n_noise: int, configs: str, providers: str, model_key: str) -> None:
    config_list = [c.strip() for c in configs.split(",") if c.strip()]
    unknown = [c for c in config_list if c not in CONFIG_CODES]
    if unknown:
        raise click.BadParameter(f"Unknown config codes: {unknown}. Valid: {CONFIG_CODES}")

    # Resolve providers (same logic as build_boh)
    import onnxruntime as ort

    available = ort.get_available_providers()
    if providers == "cpu":
        provider_list = ["CPUExecutionProvider"]
    else:
        provider_list = [p for p in cfg.DEFAULT_PROVIDER_PRIORITY if p in available]
        if not provider_list:
            provider_list = ["CPUExecutionProvider"]
    console.print(f"ONNX providers: {provider_list}")

    console.print(f"[bold]Loading {n_speech} speech + {n_noise} noise samples...[/bold]")
    speech_items = load_speech_items(n_speech)
    noise_items = load_noise_items(n_noise)
    items = speech_items + noise_items
    console.print(f"  Loaded {len(speech_items)} speech, {len(noise_items)} noise\n")

    console.print(f"[bold]Loading models...[/bold] (ASR = {model_key})")
    asr = VietnameseASR(
        model_key=model_key, num_threads=cfg.NUM_THREADS, providers=provider_list
    )
    vad = SpeechDetector()
    try:
        boh = VietnameseBoH()
        console.print(f"  ASR + VAD + BoH ({len(boh)} phrases) ready\n")
    except FileNotFoundError:
        boh = None
        console.print("  [yellow]ASR + VAD ready; BoH artifact missing → BoH configs will skip stage 4[/yellow]\n")
    robust_asr = RobustASR(asr=asr, vad=vad, boh=boh)
    console.print(
        "  RobustASR thresholds: "
        f"min_avg_logprob={robust_asr.min_avg_logprob:.3f}, "
        f"max_compression_ratio={robust_asr.max_compression_ratio:.3f}\n"
    )

    # Warmup
    if items:
        asr.transcribe(items[0].audio)

    all_results: dict[str, dict] = {}
    reports: dict[str, RobustnessReport] = {}
    for code in config_list:
        console.print(f"[bold cyan]{CONFIG_LABELS[code]}[/bold cyan]")
        run = run_config(code, items, asr, vad, boh, robust_asr)
        metrics = compute_metrics(items, run["predictions"], run["latencies_ms"])
        report = _robustness_report(run["diagnostics"])
        reports[code] = report
        # Single WER/CER definition across the summary table, saved JSON, and
        # both plot scripts: WER on *accepted* speech (what the note documents),
        # with false-reject reported as its own orthogonal axis. The all-speech
        # variant (rejected speech counted as error) is kept under explicit keys
        # so nothing is lost. When no speech survives, fall back to all-speech.
        all_results[code] = {
            **metrics,
            "wer_all_speech": metrics["wer"],
            "cer_all_speech": metrics["cer"],
            "wer": report.wer if report.wer is not None else metrics["wer"],
            "cer": report.cer if report.cer is not None else metrics["cer"],
            "robustness": _report_to_dict(report),
            "predictions": run["predictions"],
            "latencies_ms": run["latencies_ms"],
            "diagnostics": run["diagnostics"],
        }

    # Summary table
    table = Table(title=f"Table VII replication — Vietnamese {model_key}")
    table.add_column("Config", style="cyan")
    table.add_column("WER", justify="right", style="yellow")
    table.add_column("CER", justify="right")
    table.add_column("Halluc rate", justify="right", style="red")
    table.add_column("False-rej", justify="right", style="magenta")
    table.add_column("Lat p50 ms", justify="right")
    table.add_column("Lat p95 ms", justify="right")
    for code in config_list:
        m = all_results[code]
        table.add_row(
            CONFIG_LABELS[code],
            f"{m['wer'] * 100:.2f}%",
            f"{m['cer'] * 100:.2f}%",
            f"{m['hallucination_rate'] * 100:.2f}%",
            f"{reports[code].false_reject_rate * 100:.2f}%",
            f"{m['latency_p50_ms']:.0f}",
            f"{m['latency_p95_ms']:.0f}",
        )
    console.print(table)

    # Stage-contribution: which stage caught each non-speech item (full pipeline).
    if "vad_deloop_boh" in reports:
        _print_stage_breakdown(reports["vad_deloop_boh"])

    # Save. Default model keeps the canonical filename; others get their own so
    # a "robustness x model size" sweep does not overwrite itself.
    cfg.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_name = (
        "table7_replication.json"
        if model_key == DEFAULT_ASR_MODEL_KEY
        else f"table7_{model_key}.json"
    )
    out_path = cfg.EVAL_RESULTS_DIR / out_name
    payload = {
        "metadata": {
            "execution_mode": "local",
            "n_speech": len(speech_items),
            "n_noise": len(noise_items),
            "configs": config_list,
            "providers": provider_list,
            "asr_runtime_identity": asr.runtime_metadata(),
            "boh_n_phrases": len(boh) if boh else 0,
            "boh_loaded": boh is not None,
            "vad_params": {
                "threshold": vad.threshold,
                "min_speech_ms": vad.min_speech_ms,
                "min_silence_ms": vad.min_silence_ms,
                "speech_pad_ms": vad.speech_pad_ms,
            },
            "robust_asr": {
                "full_pipeline_config": "vad_deloop_boh",
                "uses_production_class": True,
                "min_avg_logprob": robust_asr.min_avg_logprob,
                "max_compression_ratio": robust_asr.max_compression_ratio,
            },
            "speech_dataset": f"{cfg.FLEURS_REPO}:{cfg.FLEURS_LANG}:{cfg.FLEURS_SPLIT}",
            "noise_dataset": "ESC-50 (filtered) + synthetic silence/white/pink",
            "created_by": "local.eval_table7",
            "created_at_utc": datetime.now(UTC).isoformat(),
        },
        "results": {
            code: {
                k: v for k, v in r.items() if k not in ("predictions", "latencies_ms")
            }
            for code, r in all_results.items()
        },
        "per_item_predictions": {
            code: r["predictions"] for code, r in all_results.items()
        },
        "per_item_diagnostics": {
            code: r["diagnostics"] for code, r in all_results.items()
        },
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"\n[green]✓ Saved {out_path}[/green]")


if __name__ == "__main__":
    main()
