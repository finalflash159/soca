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
from collections import Counter
from datetime import UTC, datetime

import click
import librosa
import numpy as np
import soundfile as sf
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


def write_wav(
    filename: str, audio: np.ndarray, source: str, label: str, subtype: str = "pure"
) -> dict:
    audio = np.asarray(audio, dtype=np.float32)
    sf.write(cfg.NOISE_WAV_DIR / filename, audio, cfg.SAMPLE_RATE)
    return {
        "path": f"wav/{filename}",
        "source": source,
        "label": label,
        "subtype": subtype,  # "pure" (VAD rejects) | "speech_like" (leaks past VAD)
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
    from datasets import load_dataset  # lazy: only the fresh-build path needs HF datasets

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


def rms(audio: np.ndarray) -> float:
    if not audio.size:
        return 0.0
    a = np.asarray(audio, dtype=np.float64)
    return float(np.sqrt(np.mean(a * a)))


def normalize_rms(audio: np.ndarray, target_rms: float) -> np.ndarray:
    """Scale to a target RMS, then guard the peak so nothing clips."""
    current = rms(audio)
    if current < 1e-8:
        return audio.astype(np.float32)
    scaled = audio * (target_rms / current)
    peak = float(np.max(np.abs(scaled)))
    if peak > 0.99:
        scaled = scaled * (0.99 / peak)
    return scaled.astype(np.float32)


def mix_babble(
    segments: list[np.ndarray],
    rng: np.random.Generator,
    target_rms: float,
    reverse_prob: float,
) -> np.ndarray:
    """Overlap equal-length voice segments into speech-like babble.

    Some segments are reversed (per ``reverse_prob``) so no single voice stays
    intelligible; the sum is then normalised to ``target_rms``. Pure function —
    fully determined by ``rng``, so callers reproduce it with a fixed seed."""
    if not segments:
        raise ValueError("mix_babble needs at least one segment")

    length = len(segments[0])
    mix = np.zeros(length, dtype=np.float64)
    for seg in segments:
        if len(seg) != length:
            raise ValueError("all babble segments must share one length")
        voice = seg[::-1] if rng.random() < reverse_prob else seg
        mix += voice.astype(np.float64) * rng.uniform(0.6, 1.0)
    return normalize_rms(mix, target_rms)


def _extract_segment(audio: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """A random window of length n_samples; tile the source if it is too short."""
    if len(audio) < n_samples:
        reps = int(np.ceil(n_samples / max(len(audio), 1)))
        audio = np.tile(audio, reps)
    start = int(rng.integers(0, len(audio) - n_samples + 1))
    return audio[start : start + n_samples]


def _load_fleurs_pool() -> list[np.ndarray]:
    if not cfg.FLEURS_MANIFEST.exists():
        return []
    pool: list[np.ndarray] = []
    with cfg.FLEURS_MANIFEST.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            audio, sr = sf.read(cfg.FLEURS_WAV_DIR / row["filename"])
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            if sr != cfg.SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=cfg.SAMPLE_RATE)
            pool.append(audio)
    return pool


def collect_synthetic_babble(
    rows: list[dict], rng: np.random.Generator, target: int | None = None
) -> None:
    """Build speech-like babble by overlapping FLEURS voices (subtype=speech_like).

    ``target`` caps the total sample count on the fresh-build path so babble
    respects ``--target``; ``--augment`` passes ``None`` (babble is additive there).
    """
    babble_cfg = cfg.SYNTHETIC_BABBLE
    if not babble_cfg["enabled"]:
        return

    pool = _load_fleurs_pool()
    max_voices = max(babble_cfg["voices_per_clip"])
    if len(pool) < max_voices:
        console.print(
            f"[yellow]Skipping babble: need >= {max_voices} FLEURS clips, found {len(pool)}. "
            "Run: uv run python -m local.download_fleurs[/yellow]"
        )
        return

    n_babble = babble_cfg["n_samples"]
    if target is not None:
        n_babble = max(0, min(n_babble, target - len(rows)))

    console.print(f"[bold]Synthesizing babble[/bold] ({n_babble} clips from FLEURS)")
    for idx in range(n_babble):
        n_voices = int(rng.choice(babble_cfg["voices_per_clip"]))
        duration = float(rng.choice(babble_cfg["durations_s"]))
        n_samples = int(duration * cfg.SAMPLE_RATE)
        picks = rng.choice(len(pool), size=n_voices, replace=False)
        segments = [_extract_segment(pool[i], n_samples, rng) for i in picks]
        babble = mix_babble(segments, rng, babble_cfg["target_rms"], babble_cfg["reverse_prob"])
        rows.append(
            write_wav(
                f"babble_{idx:04d}.wav", babble, "synthetic_babble", "babble",
                subtype="speech_like",
            )
        )
    console.print(f"  Babble collected: {n_babble}")


def _load_existing_rows() -> list[dict]:
    """Read the current manifest, backfilling subtype=pure on legacy rows."""
    rows: list[dict] = []
    with cfg.NOISE_MANIFEST.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("subtype", "pure")
            rows.append(row)
    return rows


@click.command()
@click.option("--target", default=cfg.TARGET_TOTAL_SAMPLES, type=int,
              help="Total non-speech samples to collect.")
@click.option("--seed", default=cfg.SEED, type=int, help="RNG seed for reproducibility.")
@click.option("--force/--no-force", default=False,
              help="Overwrite existing manifest even if it has enough samples.")
@click.option("--augment/--no-augment", default=False,
              help="Append only speech-like babble to the existing manifest "
                   "(reuses ESC-50 on disk; no HF datasets download).")
def main(target: int, seed: int, force: bool, augment: bool) -> None:
    cfg.NOISE_WAV_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    if augment:
        if not cfg.NOISE_MANIFEST.exists():
            raise click.ClickException(
                "--augment needs an existing manifest; run a fresh build first."
            )
        # Drop any prior babble so re-running is idempotent, then re-add.
        rows = [r for r in _load_existing_rows() if r.get("subtype") != "speech_like"]
        before = len(rows)
        collect_synthetic_babble(rows, rng)
        console.print(
            f"  Augmented {before} pure rows with {len(rows) - before} speech-like babble"
        )
    else:
        if cfg.NOISE_MANIFEST.exists() and not force:
            with cfg.NOISE_MANIFEST.open() as f:
                existing = sum(1 for line in f if line.strip())
            if existing >= target:
                console.print(
                    f"[yellow]Manifest already has {existing} >= target {target} samples. "
                    f"Use --force to rebuild, or --augment to add speech-like babble.[/yellow]"
                )
                return
        rows = []
        collect_esc50(rows, target)
        collect_synthetic(rows, target, rng)
        collect_synthetic_babble(rows, rng, target)

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
        "synthetic_babble": cfg.SYNTHETIC_BABBLE,
        "subtype_counts": dict(Counter(r.get("subtype", "pure") for r in rows)),
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
