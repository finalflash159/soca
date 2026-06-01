"""CLI: collect non-speech audio for BoH construction (local Mac).

Equivalent of notebooks/01_noise_data_collection.ipynb but runnable as:

    uv run python -m local.collect_noise --target 800

Output:
    data/noise_for_boh/wav/*.wav
    data/noise_for_boh/manifest.jsonl
    data/noise_for_boh/noise_collection_config.json
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime

import click
import librosa
import numpy as np
import soundfile as sf
from datasets import load_dataset
from rich.console import Console

from local import config as cfg

console = Console()


def pink_noise(n_samples: int, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """Generate pink noise (1/f power spectrum) at the given amplitude."""
    white = rng.normal(0.0, 1.0, n_samples)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / cfg.SAMPLE_RATE)
    spectrum = np.fft.rfft(white)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    pink = np.fft.irfft(spectrum * scale, n=n_samples)
    pink = pink / max(np.max(np.abs(pink)), 1e-8)
    return (pink * amplitude).astype(np.float32)


def write_wav(filename: str, audio: np.ndarray, source: str, label: str) -> dict:
    audio = np.asarray(audio, dtype=np.float32)
    sf.write(cfg.NOISE_WAV_DIR / filename, audio, cfg.SAMPLE_RATE)
    return {
        "path": f"wav/{filename}",
        "source": source,
        "label": label,
        "duration_s": len(audio) / cfg.SAMPLE_RATE,
        "sampling_rate": cfg.SAMPLE_RATE,
        "expected_transcript": "",
    }


def collect_esc50(rows: list[dict], target: int) -> None:
    """Stream ESC-50, resample to 16kHz mono, skip human-voice categories."""
    esc50_cfg = cfg.NOISE_SOURCES["esc50"]
    if not esc50_cfg["enabled"]:
        return

    exclude = set(esc50_cfg["exclude_categories"])
    n_target_esc = min(esc50_cfg["n_samples"], target - len(rows))
    if n_target_esc <= 0:
        return

    console.print(f"[bold]Streaming ESC-50[/bold] (target {n_target_esc} samples)")
    dataset = load_dataset(esc50_cfg["hf_repo"], split=esc50_cfg["split"], streaming=True)

    count = 0
    for ex in dataset:
        category = ex.get("category", "")
        if category in exclude:
            continue

        audio = ex["audio"]
        sr = audio["sampling_rate"]
        arr = audio["array"].astype(np.float32)
        if sr != cfg.SAMPLE_RATE:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=cfg.SAMPLE_RATE)

        rows.append(write_wav(f"esc50_{count:04d}.wav", arr, "esc50", category))
        count += 1
        if count >= n_target_esc or len(rows) >= target:
            break

    console.print(f"  ESC-50 collected: {count}")


def collect_synthetic(rows: list[dict], target: int, rng: np.random.Generator) -> None:
    """Generate synthetic silence, white noise, pink noise."""
    if not cfg.SYNTHETIC_NOISE["enabled"]:
        return

    for kind, n_items in [
        ("silence", cfg.SYNTHETIC_NOISE["n_silence"]),
        ("white_noise", cfg.SYNTHETIC_NOISE["n_white_noise"]),
        ("pink_noise", cfg.SYNTHETIC_NOISE["n_pink_noise"]),
    ]:
        before = len(rows)
        for idx in range(n_items):
            if len(rows) >= target:
                return

            duration = float(rng.choice(cfg.SYNTHETIC_NOISE["durations_s"]))
            amplitude = float(rng.choice(cfg.SYNTHETIC_NOISE["amplitudes"]))
            n_samples = int(duration * cfg.SAMPLE_RATE)

            if kind == "silence":
                arr = np.zeros(n_samples, dtype=np.float32)
            elif kind == "white_noise":
                arr = rng.normal(0.0, amplitude, n_samples).astype(np.float32)
            elif kind == "pink_noise":
                arr = pink_noise(n_samples, amplitude, rng)
            else:
                raise ValueError(f"Unsupported synthetic noise kind: {kind}")

            rows.append(write_wav(f"synthetic_{kind}_{idx:04d}.wav", arr, f"synthetic_{kind}", kind))

        console.print(f"  Synthetic {kind}: {len(rows) - before}")


@click.command()
@click.option("--target", default=cfg.TARGET_TOTAL_SAMPLES, type=int,
              help="Total non-speech samples to collect.")
@click.option("--seed", default=cfg.SEED, type=int, help="RNG seed for reproducibility.")
@click.option("--force/--no-force", default=False,
              help="Overwrite existing manifest even if it has enough samples.")
def main(target: int, seed: int, force: bool) -> None:
    cfg.NOISE_WAV_DIR.mkdir(parents=True, exist_ok=True)

    if cfg.NOISE_MANIFEST.exists() and not force:
        with cfg.NOISE_MANIFEST.open() as f:
            existing = sum(1 for line in f if line.strip())
        if existing >= target:
            console.print(
                f"[yellow]Manifest already has {existing} >= target {target} samples. "
                f"Use --force to rebuild.[/yellow]"
            )
            return

    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    collect_esc50(rows, target)
    collect_synthetic(rows, target, rng)

    with cfg.NOISE_MANIFEST.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    snapshot = {
        "project": "soca",
        "run_name": "d2_5_noise_collection_local",
        "execution_mode": "local",
        "seed": seed,
        "sample_rate": cfg.SAMPLE_RATE,
        "target_total_samples": target,
        "num_rows": len(rows),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "noise_sources": cfg.NOISE_SOURCES,
        "synthetic_noise": cfg.SYNTHETIC_NOISE,
        "manifest": str(cfg.NOISE_MANIFEST),
        "noise_dir": str(cfg.NOISE_WAV_DIR),
    }
    cfg.NOISE_CONFIG_SNAPSHOT.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    console.print(f"\n[green]✓ Collected {len(rows)} samples[/green]")
    console.print(f"  Manifest: {cfg.NOISE_MANIFEST}")
    console.print(f"  Config snapshot: {cfg.NOISE_CONFIG_SNAPSHOT}")


if __name__ == "__main__":
    main()
