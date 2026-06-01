from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
from rich.console import Console
from rich.progress import track
from rich.table import Table

from soca.asr import (
    ASR_MODEL_REGISTRY,
    DEFAULT_ASR_MODEL_KEY,
    VietnameseASR,
    get_asr_model_config,
    get_asr_profile_model_keys,
)

console = Console()
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "fleurs_vi"


def providers_from_arg(value: str) -> list[str]:
    if value == "cpu":
        return ["CPUExecutionProvider"]
    providers = ort.get_available_providers()
    if "CoreMLExecutionProvider" in providers:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def parse_model_list(values: Sequence[str]) -> list[str]:
    model_keys: list[str] = []
    for value in values:
        model_keys.extend(part.strip() for part in value.split(",") if part.strip())

    unknown = sorted(set(model_keys) - set(ASR_MODEL_REGISTRY))
    if unknown:
        valid = ", ".join(sorted(ASR_MODEL_REGISTRY))
        raise ValueError(f"Unknown ASR model key(s): {', '.join(unknown)}. Valid keys: {valid}")
    return model_keys


def select_model_keys(model_keys: Sequence[str], profile: str | None) -> list[str]:
    selected: list[str] = []
    if profile:
        selected.extend(get_asr_profile_model_keys(profile))
    selected.extend(model_keys)
    if not selected:
        selected.append(DEFAULT_ASR_MODEL_KEY)

    result: list[str] = []
    seen: set[str] = set()
    for model_key in selected:
        if model_key in seen:
            continue
        seen.add(model_key)
        result.append(model_key)
    return result


def load_manifest(data_dir: Path, limit: int | None) -> list[dict]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing manifest: {manifest_path}\n"
            "Run: uv run python eval/download_common_voice_vi.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = [entry for entry in manifest if (data_dir / entry["filename"]).exists()]
    if not manifest:
        raise FileNotFoundError(
            f"No manifest audio files found under {data_dir}\n"
            "Run: uv run --extra eval python eval/download_common_voice_vi.py"
        )
    if limit is not None:
        return manifest[:limit]
    return manifest


def run_model_eval(
    model_key: str,
    manifest: Sequence[dict],
    data_dir: Path,
    providers: list[str],
    num_threads: int,
    skip_missing: bool,
) -> dict | None:
    from jiwer import cer, wer

    config = get_asr_model_config(model_key)
    if not config.local_dir.exists():
        message = f"Missing {model_key}: {config.local_dir}\nRun: {config.download_command}"
        if skip_missing:
            console.print(f"[yellow]{message}[/yellow]")
            return None
        raise FileNotFoundError(message)

    console.print(
        f"\n[bold]Loading[/bold] {model_key} ({config.params_m}M)\n"
        f"providers={providers}"
    )
    asr = VietnameseASR(model_key=model_key, num_threads=num_threads, providers=providers)

    if manifest:
        _ = asr.transcribe_file(data_dir / manifest[0]["filename"])

    samples = []
    total_audio = 0.0
    total_latency = 0.0

    for entry in track(manifest, description=f"Transcribing {model_key}..."):
        result = asr.transcribe_file(data_dir / entry["filename"])
        samples.append(
            {
                "filename": entry["filename"],
                "ground_truth": entry["ground_truth"],
                "prediction": result.text,
                "latency_ms": result.latency_ms,
                "audio_ms": result.audio_duration_ms,
                "rtf": result.rtf,
                "avg_logprob": result.avg_logprob,
            }
        )
        total_audio += result.audio_duration_ms
        total_latency += result.latency_ms

    refs = [sample["ground_truth"].lower().strip() for sample in samples]
    hyps = [sample["prediction"].lower().strip() for sample in samples]
    latencies = [sample["latency_ms"] for sample in samples]
    rtfs = [sample["rtf"] for sample in samples]

    return {
        "model_key": model_key,
        "hf_repo": config.hf_repo,
        "params_m": config.params_m,
        "role": config.role,
        "providers": providers,
        "num_threads": num_threads,
        "num_samples": len(samples),
        "wer": wer(refs, hyps) if samples else 0.0,
        "cer": cer(refs, hyps) if samples else 0.0,
        "avg_latency_ms": total_latency / max(len(samples), 1),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "avg_rtf": total_latency / max(total_audio, 1.0),
        "p50_rtf": percentile(rtfs, 50),
        "p95_rtf": percentile(rtfs, 95),
        "samples": samples,
    }


def print_summary(reports: Sequence[dict]) -> None:
    table = Table(title="Soca PhoWhisper ASR Bakeoff")
    table.add_column("Model", style="cyan")
    table.add_column("Params", justify="right")
    table.add_column("WER", justify="right")
    table.add_column("CER", justify="right")
    table.add_column("RTF avg", justify="right")
    table.add_column("RTF p95", justify="right")
    table.add_column("Lat p50 ms", justify="right")
    table.add_column("Lat p95 ms", justify="right")

    for report in reports:
        table.add_row(
            report["model_key"],
            f"{report['params_m']}M",
            f"{report['wer']:.2%}",
            f"{report['cer']:.2%}",
            f"{report['avg_rtf']:.3f}",
            f"{report['p95_rtf']:.3f}",
            f"{report['p50_latency_ms']:.0f}",
            f"{report['p95_latency_ms']:.0f}",
        )

    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PhoWhisper ONNX ASR models.")
    parser.add_argument(
        "--model",
        action="append",
        choices=sorted(ASR_MODEL_REGISTRY),
        default=[],
        help="ASR model key. May be passed multiple times.",
    )
    parser.add_argument(
        "--models",
        action="append",
        default=[],
        help="Comma-separated ASR model keys. May be passed multiple times.",
    )
    parser.add_argument(
        "--profile",
        choices=["minimal", "bakeoff", "full"],
        default=None,
        help="Evaluate an ASR profile.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of eval samples.")
    parser.add_argument(
        "--providers",
        choices=["auto", "cpu"],
        default="auto",
        help="ONNX Runtime provider policy.",
    )
    parser.add_argument("--num-threads", type=int, default=4, help="ONNX Runtime thread count.")
    parser.add_argument("--skip-missing", action="store_true", help="Skip models not downloaded yet.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="FLEURS eval data dir.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "eval" / "results",
        help="Directory for JSON output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    explicit_models = [*args.model, *parse_model_list(args.models)]
    model_keys = select_model_keys(explicit_models, args.profile)
    providers = providers_from_arg(args.providers)
    manifest = load_manifest(args.data_dir, args.limit)

    console.print(f"Loaded manifest with {len(manifest)} samples")

    reports: list[dict] = []
    for model_key in model_keys:
        report = run_model_eval(
            model_key=model_key,
            manifest=manifest,
            data_dir=args.data_dir,
            providers=providers,
            num_threads=args.num_threads,
            skip_missing=args.skip_missing,
        )
        if report is not None:
            reports.append(report)

    if not reports:
        console.print("[yellow]No ASR reports were produced; all selected models were skipped.[/yellow]")
        return

    print_summary(reports)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "profile_" + args.profile if args.profile else "_".join(model_keys)
    if args.limit:
        suffix += f"_n{args.limit}"
    output_path = args.output_dir / f"asr_bakeoff_{suffix}_{timestamp}.json"
    output_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"\n[green]Saved -> {output_path}[/green]")

    if model_keys == [DEFAULT_ASR_MODEL_KEY] and args.limit is None:
        baseline_path = args.output_dir / "asr_d1_baseline.json"
        baseline_path.write_text(json.dumps(reports[0], ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"[green]Updated -> {baseline_path}[/green]")


if __name__ == "__main__":
    main()
