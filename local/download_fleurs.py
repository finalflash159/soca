"""CLI: download FLEURS vi_vn speech samples for calibration + benchmark.

FLEURS = "Few-shot Learning Evaluation of Universal Representations of Speech"
(Conneau et al., Google). Used here because:
  - Native Vietnamese speech with verified transcripts
  - Already 16kHz mono
  - Public, no auth needed
  - Mix of speakers/dialects → realistic for command recognition

Output:
    data/fleurs_vi/wav/*.wav
    data/fleurs_vi/manifest.jsonl  — each row: {filename, ground_truth, duration_s, speaker_gender}

Usage:
    uv run python -m local.download_fleurs --target 200
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import click
import librosa
import numpy as np
import soundfile as sf
from datasets import load_dataset
from rich.console import Console

from local import config as cfg

console = Console()


@click.command()
@click.option(
    "--target", default=cfg.FLEURS_DEFAULT_TARGET, type=int,
    help="Number of speech samples to download.",
)
@click.option("--split", default=cfg.FLEURS_SPLIT,
              help="FLEURS split: test, validation, train.")
@click.option("--force/--no-force", default=False,
              help="Re-download even if manifest is already large enough.")
def main(target: int, split: str, force: bool) -> None:
    cfg.FLEURS_WAV_DIR.mkdir(parents=True, exist_ok=True)

    if cfg.FLEURS_MANIFEST.exists() and not force:
        with cfg.FLEURS_MANIFEST.open() as f:
            existing = sum(1 for line in f if line.strip())
        if existing >= target:
            console.print(
                f"[yellow]Manifest already has {existing} >= target {target}. "
                f"Use --force to rebuild.[/yellow]"
            )
            return

    console.print(f"[bold]Streaming FLEURS {cfg.FLEURS_LANG} split={split} (target {target})[/bold]")
    dataset = load_dataset(cfg.FLEURS_REPO, cfg.FLEURS_LANG, split=split, streaming=True)

    rows: list[dict] = []
    for idx, ex in enumerate(dataset):
        if len(rows) >= target:
            break

        audio = ex["audio"]
        sr = audio["sampling_rate"]
        arr = audio["array"].astype(np.float32)
        if sr != cfg.SAMPLE_RATE:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=cfg.SAMPLE_RATE)

        # FLEURS exposes `raw_transcription` (with casing/punctuation) and
        # `transcription` (normalized). Use raw for display; normalize at WER time.
        ground_truth = ex.get("raw_transcription") or ex.get("transcription", "")
        if not ground_truth.strip():
            continue

        filename = f"fleurs_vi_{idx:05d}.wav"
        sf.write(cfg.FLEURS_WAV_DIR / filename, arr, cfg.SAMPLE_RATE)
        rows.append(
            {
                "filename": filename,
                "ground_truth": ground_truth.strip(),
                "duration_s": len(arr) / cfg.SAMPLE_RATE,
                "sampling_rate": cfg.SAMPLE_RATE,
                "gender": ex.get("gender", -1),
                "fleurs_id": ex.get("id"),
            }
        )

    with cfg.FLEURS_MANIFEST.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    snapshot = {
        "project": "soca",
        "run_name": "fleurs_vi_download_local",
        "dataset": cfg.FLEURS_REPO,
        "language": cfg.FLEURS_LANG,
        "split": split,
        "num_rows": len(rows),
        "target": target,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    (cfg.FLEURS_DIR / "fleurs_download_config.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    console.print(f"\n[green]✓ Downloaded {len(rows)} FLEURS {cfg.FLEURS_LANG} samples[/green]")
    console.print(f"  Manifest: {cfg.FLEURS_MANIFEST}")


if __name__ == "__main__":
    main()
