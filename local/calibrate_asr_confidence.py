"""CLI: calibrate ASR confidence thresholds from speech + non-speech audio.

= notebooks/05_asr_confidence_calibration.ipynb (Mac CLI version).

This script calibrates the model-confidence guard used by `RobustASR`:

    VAD -> PhoWhisper -> avg_logprob / compression_ratio guard

Why this exists:
    `local/calibrate_thresholds.py` calibrates text-only heuristics from
    ground-truth transcripts. This script calibrates ASR-runtime confidence
    signals from actual model outputs:

    - avg_logprob: lower means the decoder was less confident
    - compression_ratio: higher often means repetitive loop text

The calibration must be tied to the exact ASR runtime identity:
ONNX files, decoder variant, greedy decode, max_new_tokens, providers, and
VAD parameters. Changing any of those can shift the metric distribution.

Outputs:
    - eval/results/asr_confidence_calibration_{model_key}.json
      Full per-sample audit log + recommended thresholds for each model.

    - data/asr/threshold_calibration.json
      Merged calibration payload under `asr_confidence_by_model`.

Usage:
    # one-time prerequisites
    uv run python -m local.download_fleurs --target 200
    uv run python -m local.collect_noise

    # smoke run
    uv run python -m local.calibrate_asr_confidence --model phowhisper_base --n-speech 5 --n-noise 5 --providers cpu

    # main local run on Mac
    uv run python -m local.calibrate_asr_confidence --model phowhisper_base --model phowhisper_small --n-speech 200 --n-noise 50
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
from rich.console import Console
from rich.progress import track
from rich.table import Table

from local import config as cfg
from soca.asr import SpeechDetector, VietnameseASR
from soca.asr.hallucination_heuristics import compression_ratio
from soca.asr.registry import DEFAULT_ASR_MODEL_KEY

console = Console()


@dataclass(frozen=True)
class CalibrationItem:
    """One audio item used for confidence calibration."""

    path: Path
    kind: str  # "speech" | "noise"
    ground_truth: str
    source: str
    label: str


def load_audio(path: Path) -> np.ndarray:
    """Load audio as 16kHz mono float32, matching the ASR runtime contract."""
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != cfg.SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=cfg.SAMPLE_RATE)
    return audio.astype(np.float32, copy=False)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_speech_items(n: int) -> list[CalibrationItem]:
    if not cfg.FLEURS_MANIFEST.exists():
        raise click.ClickException(
            f"FLEURS manifest missing: {cfg.FLEURS_MANIFEST}\n"
            "Run: uv run python -m local.download_fleurs --target 200"
        )

    items: list[CalibrationItem] = []
    for row in read_jsonl(cfg.FLEURS_MANIFEST):
        items.append(
            CalibrationItem(
                path=cfg.FLEURS_WAV_DIR / row["filename"],
                kind="speech",
                ground_truth=row.get("ground_truth", ""),
                source="fleurs",
                label=row.get("speaker_id", ""),
            )
        )
        if len(items) >= n:
            break
    return items


def load_noise_items(n: int) -> list[CalibrationItem]:
    if not cfg.NOISE_MANIFEST.exists():
        raise click.ClickException(
            f"Noise manifest missing: {cfg.NOISE_MANIFEST}\n"
            "Run: uv run python -m local.collect_noise"
        )

    items: list[CalibrationItem] = []
    for row in read_jsonl(cfg.NOISE_MANIFEST):
        items.append(
            CalibrationItem(
                path=cfg.NOISE_ROOT / row["path"],
                kind="noise",
                ground_truth="",
                source=row.get("source", "noise"),
                label=row.get("label", ""),
            )
        )
        if len(items) >= n:
            break
    return items


def resolve_providers(mode: str) -> list[str]:
    """Resolve ONNX providers for Mac/local execution."""
    if mode == "cpu":
        return ["CPUExecutionProvider"]

    available = ort.get_available_providers()
    providers = [p for p in cfg.DEFAULT_PROVIDER_PRIORITY if p in available]
    return providers or ["CPUExecutionProvider"]


def percentile_summary(values: list[float], metric: str, cohort: str) -> dict[str, Any]:
    """Return stable percentile stats. Empty inputs are represented explicitly."""
    if not values:
        return {
            "metric": metric,
            "cohort": cohort,
            "n": 0,
            "mean": None,
            "min": None,
            "p01": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    arr = np.array(values, dtype=np.float64)
    return {
        "metric": metric,
        "cohort": cohort,
        "n": int(len(arr)),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "p01": float(np.percentile(arr, 1)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def fmt_optional(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def run_item(
    item: CalibrationItem,
    asr: VietnameseASR,
    vad: SpeechDetector,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Run VAD and optional ASR for one calibration item."""
    audio = load_audio(item.path)
    duration_ms = len(audio) / cfg.SAMPLE_RATE * 1000

    vad_result = vad.detect(audio)
    row: dict[str, Any] = {
        "kind": item.kind,
        "path": str(item.path),
        "filename": item.path.name,
        "source": item.source,
        "label": item.label,
        "ground_truth": item.ground_truth,
        "duration_ms": duration_ms,
        "vad_has_speech": vad_result.has_speech,
        "speech_duration_ms": vad_result.speech_duration_ms,
        "speech_ratio": vad_result.speech_ratio,
        "vad_latency_ms": vad_result.vad_latency_ms,
        "n_speech_segments": vad_result.n_speech_segments,
        "asr_ran": False,
        "raw_text": "",
        "avg_logprob": None,
        "compression_ratio": 0.0,
        "asr_latency_ms": 0.0,
        "rtf": 0.0,
    }

    if not vad_result.has_speech:
        return row

    asr_result = asr.transcribe(vad_result.speech_audio, max_new_tokens=max_new_tokens)
    row.update(
        {
            "asr_ran": True,
            "raw_text": asr_result.text,
            "avg_logprob": asr_result.avg_logprob,
            "compression_ratio": compression_ratio(asr_result.text),
            "asr_latency_ms": asr_result.latency_ms,
            "rtf": asr_result.rtf,
        }
    )
    return row


def recommend_avg_logprob_threshold(
    speech_values: list[float],
    noise_values: list[float],
    fallback: float,
) -> dict[str, Any]:
    """Pick a conservative lower bound for accepted ASR confidence.

    Interpretation:
        - Speech should usually have higher avg_logprob.
        - Noise/hallucination should usually have lower avg_logprob.

    If noise max is below speech p01, there is a clean gap, so use the midpoint.
    If distributions overlap, protect real speech first and use speech p01.
    """
    if not speech_values:
        return {
            "value": fallback,
            "rule": "fallback_no_speech_asr_rows",
            "speech_p01": None,
            "noise_max": max(noise_values) if noise_values else None,
            "clean_separation": False,
        }

    speech_p01 = float(np.percentile(np.array(speech_values), 1))
    if not noise_values:
        return {
            "value": speech_p01,
            "rule": "speech_p01_noise_vad_skipped",
            "speech_p01": speech_p01,
            "noise_max": None,
            "clean_separation": True,
        }

    noise_max = float(max(noise_values))
    clean_separation = noise_max < speech_p01
    if clean_separation:
        value = (noise_max + speech_p01) / 2
        rule = "midpoint_between_noise_max_and_speech_p01"
    else:
        value = speech_p01
        rule = "speech_p01_distribution_overlap"

    return {
        "value": value,
        "rule": rule,
        "speech_p01": speech_p01,
        "noise_max": noise_max,
        "clean_separation": clean_separation,
    }


def recommend_compression_threshold(speech_values: list[float], fallback: float) -> dict[str, Any]:
    """Pick max compression ratio threshold from real speech distribution."""
    if not speech_values:
        return {
            "value": fallback,
            "rule": "fallback_no_speech_asr_rows",
            "speech_p99": None,
        }

    speech_p99 = float(np.percentile(np.array(speech_values), 99))
    # 2.4 is the common Whisper default. Keep it if it already clears p99.
    if speech_p99 <= fallback:
        return {
            "value": fallback,
            "rule": "whisper_default_above_speech_p99",
            "speech_p99": speech_p99,
        }

    return {
        "value": speech_p99 * 1.15,
        "rule": "speech_p99_plus_15_percent",
        "speech_p99": speech_p99,
    }


def print_dataset_table(rows: list[dict[str, Any]]) -> None:
    table = Table(title="Calibration dataset summary")
    table.add_column("Cohort", style="cyan")
    table.add_column("Items", justify="right")
    table.add_column("VAD speech", justify="right")
    table.add_column("ASR ran", justify="right")
    table.add_column("Non-empty ASR", justify="right")

    for kind in ("speech", "noise"):
        cohort = [r for r in rows if r["kind"] == kind]
        vad_speech = sum(1 for r in cohort if r["vad_has_speech"])
        asr_ran = sum(1 for r in cohort if r["asr_ran"])
        non_empty = sum(1 for r in cohort if r["raw_text"].strip())
        table.add_row(
            kind,
            str(len(cohort)),
            f"{vad_speech} ({vad_speech / max(len(cohort), 1):.1%})",
            f"{asr_ran} ({asr_ran / max(len(cohort), 1):.1%})",
            f"{non_empty} ({non_empty / max(len(cohort), 1):.1%})",
        )
    console.print(table)


def print_metric_table(stats: list[dict[str, Any]]) -> None:
    table = Table(title="ASR confidence metric distributions")
    table.add_column("Metric", style="cyan")
    table.add_column("Cohort")
    table.add_column("N", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("P01", justify="right")
    table.add_column("P05", justify="right")
    table.add_column("P50", justify="right")
    table.add_column("P95", justify="right")
    table.add_column("P99", justify="right")
    table.add_column("Max", justify="right")

    for stat in stats:
        table.add_row(
            stat["metric"],
            stat["cohort"],
            str(stat["n"]),
            fmt_optional(stat["mean"]),
            fmt_optional(stat["p01"]),
            fmt_optional(stat["p05"]),
            fmt_optional(stat["p50"]),
            fmt_optional(stat["p95"]),
            fmt_optional(stat["p99"]),
            fmt_optional(stat["max"]),
        )
    console.print(table)


def merge_threshold_file(
    model_key: str,
    asr_confidence_payload: dict[str, Any],
    *,
    update_legacy_singleton: bool = False,
) -> None:
    """Merge ASR-confidence calibration into the shared threshold JSON."""
    cfg.ASR_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if cfg.THRESHOLD_CALIBRATION_PATH.exists():
        payload = json.loads(cfg.THRESHOLD_CALIBRATION_PATH.read_text(encoding="utf-8"))
    else:
        payload = {}

    by_model = payload.setdefault("asr_confidence_by_model", {})
    if not isinstance(by_model, dict):
        by_model = {}
        payload["asr_confidence_by_model"] = by_model
    by_model[model_key] = asr_confidence_payload

    # Keep the old singleton key only as a compatibility alias. The runtime
    # loader accepts it only when its model identity matches the requested model.
    if update_legacy_singleton:
        payload["asr_confidence"] = asr_confidence_payload

    cfg.THRESHOLD_CALIBRATION_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def calibrate_model(
    *,
    model_key: str,
    n_speech: int,
    n_noise: int,
    provider_list: list[str],
    max_new_tokens: int,
    fallback_min_avg_logprob: float,
    fallback_max_compression_ratio: float,
) -> dict[str, Any]:
    speech_items = load_speech_items(n_speech)
    noise_items = load_noise_items(n_noise)
    items = speech_items + noise_items
    console.rule(f"Calibrating ASR confidence: {model_key}")
    console.print(
        f"Loaded [bold]{len(speech_items)}[/bold] speech + "
        f"[bold]{len(noise_items)}[/bold] noise samples"
    )

    console.print(f"[bold]Loading ASR + VAD...[/bold] model={model_key}")
    asr = VietnameseASR(
        model_key=model_key,
        num_threads=cfg.NUM_THREADS,
        providers=provider_list,
    )
    vad = SpeechDetector()
    runtime_identity = {
        "asr": asr.runtime_metadata(max_new_tokens=max_new_tokens),
        "vad": {
            "threshold": vad.threshold,
            "min_speech_ms": vad.min_speech_ms,
            "min_silence_ms": vad.min_silence_ms,
            "speech_pad_ms": vad.speech_pad_ms,
        },
    }

    rows: list[dict[str, Any]] = []
    for item in track(items, description="Calibrating ASR confidence"):
        rows.append(run_item(item, asr=asr, vad=vad, max_new_tokens=max_new_tokens))

    speech_asr = [r for r in rows if r["kind"] == "speech" and r["asr_ran"]]
    noise_asr = [r for r in rows if r["kind"] == "noise" and r["asr_ran"]]
    speech_avg_logprob = [float(r["avg_logprob"]) for r in speech_asr if r["avg_logprob"] is not None]
    noise_avg_logprob = [float(r["avg_logprob"]) for r in noise_asr if r["avg_logprob"] is not None]
    speech_compression = [float(r["compression_ratio"]) for r in speech_asr]
    noise_compression = [float(r["compression_ratio"]) for r in noise_asr]

    stats = [
        percentile_summary(speech_avg_logprob, "avg_logprob", "speech"),
        percentile_summary(noise_avg_logprob, "avg_logprob", "noise"),
        percentile_summary(speech_compression, "compression_ratio", "speech"),
        percentile_summary(noise_compression, "compression_ratio", "noise"),
    ]

    avg_logprob_rec = recommend_avg_logprob_threshold(
        speech_avg_logprob,
        noise_avg_logprob,
        fallback=fallback_min_avg_logprob,
    )
    compression_rec = recommend_compression_threshold(
        speech_compression,
        fallback=fallback_max_compression_ratio,
    )

    recommended = {
        "min_avg_logprob": avg_logprob_rec["value"],
        "max_compression_ratio": compression_rec["value"],
    }

    print_dataset_table(rows)
    print_metric_table(stats)

    threshold_table = Table(title="Recommended RobustASR confidence thresholds")
    threshold_table.add_column("Threshold", style="cyan")
    threshold_table.add_column("Value", justify="right", style="green")
    threshold_table.add_column("Selection rule")
    threshold_table.add_row(
        "min_avg_logprob",
        f"{recommended['min_avg_logprob']:.3f}",
        avg_logprob_rec["rule"],
    )
    threshold_table.add_row(
        "max_compression_ratio",
        f"{recommended['max_compression_ratio']:.3f}",
        compression_rec["rule"],
    )
    console.print(threshold_table)

    created_at = datetime.now(UTC).isoformat()
    calibration_payload = {
        "model_key": model_key,
        "dataset": {
            "speech": f"{cfg.FLEURS_REPO}:{cfg.FLEURS_LANG}:{cfg.FLEURS_SPLIT}",
            "noise": "data/noise_for_boh manifest",
            "n_speech_requested": n_speech,
            "n_noise_requested": n_noise,
            "n_speech_loaded": len(speech_items),
            "n_noise_loaded": len(noise_items),
        },
        "runtime_identity": runtime_identity,
        "stats": stats,
        "recommended_thresholds": recommended,
        "threshold_selection": {
            "avg_logprob": avg_logprob_rec,
            "compression_ratio": compression_rec,
        },
        "created_by": "local.calibrate_asr_confidence",
        "created_at_utc": created_at,
        "usage_note": (
            "Use recommended_thresholds as RobustASR(min_avg_logprob=..., "
            "max_compression_ratio=...). Re-run after changing model, decoder, "
            "provider, max_new_tokens, or VAD parameters."
        ),
    }

    cfg.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = cfg.EVAL_RESULTS_DIR / f"asr_confidence_calibration_{model_key}.json"
    out_payload = {
        "metadata": calibration_payload,
        "rows": rows,
    }
    out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    merge_threshold_file(
        model_key,
        calibration_payload,
        update_legacy_singleton=(model_key == DEFAULT_ASR_MODEL_KEY),
    )

    console.print(f"\n[green]✓ Saved raw calibration log -> {out_path}[/green]")
    console.print(f"[green]✓ Updated shared threshold file -> {cfg.THRESHOLD_CALIBRATION_PATH}[/green]")
    return calibration_payload


@click.command()
@click.option(
    "--model",
    "model_keys",
    multiple=True,
    default=(DEFAULT_ASR_MODEL_KEY,),
    type=click.Choice(list(cfg.MODEL_REGISTRY.keys())),
    help="ASR model(s) to calibrate. Repeat flag for multi-model run.",
)
@click.option("--n-speech", default=200, type=int, help="Number of FLEURS speech samples.")
@click.option("--n-noise", default=50, type=int, help="Number of non-speech samples.")
@click.option(
    "--providers",
    default="auto",
    type=click.Choice(["auto", "cpu"]),
    help="auto = CoreML + CPU fallback on Mac; cpu = force CPU.",
)
@click.option(
    "--max-new-tokens",
    default=cfg.MAX_NEW_TOKENS,
    type=int,
    help="PhoWhisper greedy decode cap. Must match runtime/eval settings.",
)
@click.option(
    "--fallback-min-avg-logprob",
    default=-0.25,
    type=float,
    help="Fallback when speech calibration cannot run.",
)
@click.option(
    "--fallback-max-compression-ratio",
    default=2.4,
    type=float,
    help="Fallback Whisper-style compression ratio threshold.",
)
def main(
    model_keys: tuple[str, ...],
    n_speech: int,
    n_noise: int,
    providers: str,
    max_new_tokens: int,
    fallback_min_avg_logprob: float,
    fallback_max_compression_ratio: float,
) -> None:
    provider_list = resolve_providers(providers)
    console.print(f"ONNX providers: {provider_list}")

    seen: set[str] = set()
    selected_model_keys = [key for key in model_keys if not (key in seen or seen.add(key))]
    for model_key in selected_model_keys:
        calibrate_model(
            model_key=model_key,
            n_speech=n_speech,
            n_noise=n_noise,
            provider_list=provider_list,
            max_new_tokens=max_new_tokens,
            fallback_min_avg_logprob=fallback_min_avg_logprob,
            fallback_max_compression_ratio=fallback_max_compression_ratio,
        )


if __name__ == "__main__":
    main()
