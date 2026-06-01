"""CLI: build Vietnamese Bag-of-Hallucinations from PhoWhisper outputs on noise.

Equivalent of notebooks/02_build_vietnamese_boh.ipynb but runnable as:

    uv run python -m local.build_boh
    uv run python -m local.build_boh --model phowhisper_tiny --max-files 100
    uv run python -m local.build_boh --providers cpu  # disable CoreML

Output:
    data/asr/boh/{model_key}_vi_boh_v1.json    — model-specific BoH
    data/asr/vi_boh_v1.json                    — alias for runtime model
    notebooks/outputs/{RUN_ID}/logs/boh_runs/{model_key}/phowhisper_noise_outputs.jsonl
    notebooks/outputs/{RUN_ID}/config_snapshot.json
"""

from __future__ import annotations

import json
import re
import shutil
import time
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import click
import librosa
import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.progress import track
from rich.table import Table

from local import config as cfg
from soca.asr import VietnameseASR

console = Console()


def normalize_transcript(text: str) -> str:
    """NFC + lowercase + collapse whitespace + strip boundary punctuation."""
    text = unicodedata.normalize("NFC", text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n\r.,!?;:\"'“”‘’()[]{}")


def resolve_providers(preference: str) -> list[str]:
    """Map user preference to onnxruntime providers list, with availability check."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    if preference == "cpu":
        return ["CPUExecutionProvider"]

    selected = [p for p in cfg.DEFAULT_PROVIDER_PRIORITY if p in available]
    if not selected:
        selected = ["CPUExecutionProvider"]
    console.print(f"ONNX providers available: {available}")
    console.print(f"ONNX providers selected: {selected}")
    return selected


def ensure_model(model_key: str) -> Path:
    """Download model files if missing. Return local model dir."""
    if model_key not in cfg.MODEL_REGISTRY:
        raise click.BadParameter(
            f"Unknown model_key '{model_key}'. Available: {list(cfg.MODEL_REGISTRY)}"
        )

    entry = cfg.MODEL_REGISTRY[model_key]
    model_dir = cfg.MODELS_DIR / entry["local_subdir"]
    required = [
        model_dir / "onnx" / "encoder_model.onnx",
        model_dir / "onnx" / "decoder_model.onnx",
        model_dir / "generation_config.json",
    ]
    if all(p.exists() for p in required):
        console.print(f"[dim]Model already cached: {model_key} -> {model_dir}[/dim]")
        return model_dir

    console.print(f"[bold]Downloading {entry['repo_id']} -> {model_dir}[/bold]")
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=entry["repo_id"],
        local_dir=str(model_dir),
        allow_patterns=cfg.MODEL_ALLOW_PATTERNS,
    )
    return model_dir


def load_manifest(max_files: int | None) -> list[tuple[Path, dict]]:
    if not cfg.NOISE_MANIFEST.exists():
        raise click.ClickException(
            f"Manifest not found: {cfg.NOISE_MANIFEST}\n"
            "Run: uv run python -m local.collect_noise"
        )

    rows = []
    with cfg.NOISE_MANIFEST.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    items: list[tuple[Path, dict]] = []
    missing: list[str] = []
    for row in rows:
        path = cfg.NOISE_ROOT / row["path"]
        if path.exists():
            items.append((path, row))
        else:
            missing.append(str(path))

    if max_files is not None:
        items = items[:max_files]
    if not items:
        raise click.ClickException("No audio files found. Re-check noise collection step.")

    console.print(f"Manifest rows: {len(rows)}, usable: {len(items)}, missing: {len(missing)}")
    return items


def build_boh(
    model_key: str,
    items: list[tuple[Path, dict]],
    providers: list[str],
    max_new_tokens: int,
    run_id: str,
) -> dict:
    model_dir = ensure_model(model_key)
    repo_id = cfg.MODEL_REGISTRY[model_key]["repo_id"]

    console.rule(f"Building BoH for {model_key}")
    asr = VietnameseASR(
        model_key=model_key,
        model_dir=model_dir,
        num_threads=cfg.NUM_THREADS,
        providers=providers,
    )
    runtime_identity = asr.runtime_metadata(max_new_tokens=max_new_tokens)

    log_dir = cfg.LOCAL_OUTPUTS_DIR / run_id / "logs" / "boh_runs" / model_key
    log_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl = log_dir / "phowhisper_noise_outputs.jsonl"

    outputs: list[str] = []
    output_examples: dict[str, list[str]] = {}
    n_errors = 0

    t0 = time.perf_counter()
    with raw_jsonl.open("w", encoding="utf-8") as f:
        for path, row in track(items, description=f"{model_key}: ASR on non-speech"):
            record: dict = {
                "model_key": model_key,
                "model_repo": repo_id,
                "path": row["path"],
                "source": row.get("source", ""),
                "label": row.get("label", ""),
            }
            try:
                audio, sr = sf.read(path)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                audio = audio.astype(np.float32)
                if sr != asr.SAMPLING_RATE:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=asr.SAMPLING_RATE)

                result = asr.transcribe(audio, max_new_tokens=max_new_tokens)
                text = result.text.strip()
                normalized = normalize_transcript(text)
                record.update(result.to_dict())
                record["text"] = text
                record["normalized_text"] = normalized
                record["error"] = ""
                if normalized:
                    outputs.append(normalized)
                    examples = output_examples.setdefault(normalized, [])
                    if text and text not in examples and len(examples) < 5:
                        examples.append(text)
            except Exception as exc:  # noqa: BLE001 — keep going on per-file failures
                record["text"] = ""
                record["normalized_text"] = ""
                record["error"] = repr(exc)
                n_errors += 1
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    elapsed_s = time.perf_counter() - t0
    counter = Counter(outputs)
    top_phrases = counter.most_common(100)
    total_runs = len(items)
    total_hallucinations = len(outputs)
    halluc_rate = total_hallucinations / max(total_runs, 1)

    table = Table(title=f"Top hallucinations: {model_key}")
    table.add_column("#", justify="right")
    table.add_column("Count", justify="right", style="yellow")
    table.add_column("Phrase", style="cyan")
    for idx, (phrase, count) in enumerate(top_phrases[:20], start=1):
        display = phrase if len(phrase) <= 100 else phrase[:97] + "..."
        table.add_row(str(idx), str(count), display)
    console.print(table)

    boh_candidates = [
        {
            "phrase": phrase,
            "count": count,
            "length": len(phrase),
            "keep": True,
            "examples": output_examples.get(phrase, [phrase]),
        }
        for phrase, count in top_phrases
        if count >= cfg.MIN_COUNT and len(phrase) >= cfg.MIN_CHARS
    ]

    artifact_name = f"{model_key}_vi_boh_v1.json"
    payload = {
        "metadata": {
            "model_key": model_key,
            "model_repo": repo_id,
            "model_dir": str(model_dir),
            "artifact_name": artifact_name,
            "runtime_identity": runtime_identity,
            "execution_mode": "local",
            "num_noise_samples": total_runs,
            "num_non_empty_outputs": total_hallucinations,
            "num_errors": n_errors,
            "hallucination_rate": halluc_rate,
            "selection_rule": f"count >= {cfg.MIN_COUNT} and len >= {cfg.MIN_CHARS} chars before manual review",
            "normalization": "NFC + lowercase + whitespace collapse + boundary punctuation strip",
            "created_by": "local.build_boh",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "raw_output_jsonl": str(raw_jsonl),
            "providers": providers,
            "elapsed_s": elapsed_s,
        },
        "boh": boh_candidates,
        "all_unique_outputs_with_count": [
            {"phrase": phrase, "count": count, "examples": output_examples.get(phrase, [phrase])}
            for phrase, count in top_phrases
        ],
    }

    cfg.BOH_DIR.mkdir(parents=True, exist_ok=True)
    boh_path = cfg.BOH_DIR / artifact_name
    boh_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    console.print(
        f"\n[green]✓ {model_key}[/green]: "
        f"{total_hallucinations}/{total_runs} non-empty outputs "
        f"(rate {halluc_rate:.2%}), {len(boh_candidates)} BoH candidates, "
        f"{n_errors} errors, elapsed {elapsed_s:.0f}s"
    )
    console.print(f"  Raw log: {raw_jsonl}")
    console.print(f"  BoH:     {boh_path}")
    return {"payload": payload, "boh_path": boh_path}


@click.command()
@click.option(
    "--model", "model_keys", multiple=True, default=("phowhisper_tiny",),
    type=click.Choice(list(cfg.MODEL_REGISTRY.keys())),
    help="Model(s) to process. Repeat flag for multi-model run.",
)
@click.option(
    "--runtime-model", default="phowhisper_tiny",
    type=click.Choice(list(cfg.MODEL_REGISTRY.keys())),
    help="Which model's BoH is copied to data/asr/vi_boh_v1.json for runtime loading.",
)
@click.option(
    "--max-files", default=None, type=int,
    help="Limit to first N noise files (smoke run). Default: all.",
)
@click.option(
    "--providers", default="auto",
    type=click.Choice(["auto", "cpu"]),
    help="auto = CoreML + CPU fallback (Mac), cpu = force CPU only.",
)
@click.option(
    "--max-new-tokens", default=cfg.MAX_NEW_TOKENS, type=int,
    help="Cap on tokens per ASR call. Higher catches longer hallucination loops but slower.",
)
def main(
    model_keys: tuple[str, ...],
    runtime_model: str,
    max_files: int | None,
    providers: str,
    max_new_tokens: int,
) -> None:
    if runtime_model not in model_keys:
        raise click.BadParameter(
            f"--runtime-model '{runtime_model}' must be one of --model values {model_keys}"
        )

    run_id = "d2_5_build_vietnamese_boh_local_" + datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    provider_list = resolve_providers(providers)
    items = load_manifest(max_files)

    snapshot_dir = cfg.LOCAL_OUTPUTS_DIR / run_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / "config_snapshot.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "execution_mode": "local",
                "selected_model_keys": list(model_keys),
                "runtime_model_key": runtime_model,
                "max_files": max_files,
                "max_new_tokens": max_new_tokens,
                "providers_preference": providers,
                "providers_resolved": provider_list,
                "num_threads": cfg.NUM_THREADS,
                "min_count": cfg.MIN_COUNT,
                "min_chars": cfg.MIN_CHARS,
                "decode_strategy": cfg.ASR_DECODE_STRATEGY,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    runtime_boh_path: Path | None = None
    for model_key in model_keys:
        result = build_boh(model_key, items, provider_list, max_new_tokens, run_id)
        if model_key == runtime_model:
            runtime_boh_path = result["boh_path"]

    if runtime_boh_path is not None:
        cfg.RUNTIME_BOH_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(runtime_boh_path, cfg.RUNTIME_BOH_PATH)
        console.print(
            f"\n[green]✓ Runtime alias copied:[/green] "
            f"{runtime_boh_path.name} -> {cfg.RUNTIME_BOH_PATH}"
        )


if __name__ == "__main__":
    main()
