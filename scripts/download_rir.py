"""Download the MIT Acoustic Impulse Response Survey → 16k mono RIR wavs.

MIT IR Survey (Traer & McDermott, PNAS 2016; CC-BY) measured 270 real room impulse
responses at ~1.5m source-mic spacing — the conversational speaker↔mic distance we
need to synthesize realistic echo for the P3.1 barge-in tier. BUT ReverbDB's finding
that a few well-chosen *real* RIRs beat many synthetic ones is why we use these rather
than an alpha·delay toy echo.

    uv run --with pyarrow python scripts/download_rir.py

Output:
    data/rir/mit/rir_XXXX.wav        (16k mono, peak-normalised)
    data/rir/mit/manifest.jsonl      (id, location, detail, duration)
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import click
import librosa
import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
from rich.console import Console

_REPO = "benjamin-paine/mit-impulse-response-survey"
_PARQUET = "data/train-00000-of-00001.parquet"
_SAMPLE_RATE = 16000
_OUT_DIR = Path("data/rir/mit")

console = Console()


def _decode_to_mono16k(audio_bytes: bytes) -> np.ndarray:
    arr, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != _SAMPLE_RATE:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=_SAMPLE_RATE)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 1e-8:
        arr = arr / peak  # peak-normalise so convolution gain is set by alpha, not the IR
    return np.ascontiguousarray(arr, dtype=np.float32)


@click.command()
@click.option("--limit", default=0, type=int, help="Max IRs to extract (0 = all).")
def main(limit: int) -> None:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise click.ClickException(
            "pyarrow needed to read the parquet. Run:\n"
            "  uv run --with pyarrow python scripts/download_rir.py"
        ) from exc

    console.print(f"[bold]Downloading MIT IR parquet[/bold] from {_REPO}")
    parquet_path = hf_hub_download(_REPO, _PARQUET, repo_type="dataset")
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    if limit:
        rows = rows[:limit]

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for idx, row in enumerate(rows):
        ir = _decode_to_mono16k(row["audio"]["bytes"])
        filename = f"rir_{idx:04d}.wav"
        sf.write(_OUT_DIR / filename, ir, _SAMPLE_RATE)
        manifest.append(
            {
                "filename": filename,
                "mit_id": row.get("id"),
                "location": row.get("location"),
                "detail": row.get("detail"),
                "duration_s": len(ir) / _SAMPLE_RATE,
            }
        )

    manifest_path = _OUT_DIR / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    console.print(f"[green]✓ Extracted {len(manifest)} RIRs → {_OUT_DIR}[/green]")
    console.print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
