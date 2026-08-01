"""Record the code-switch test set: press Enter to start, Enter to stop.

    uv run python local/record_codeswitch.py

Writes data/asr_codeswitch/wav/*.wav + manifest.jsonl. Safe to re-run:
sentences with an existing wav are skipped (use --redo to re-record).
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from local.codeswitch_text import english_indices, tokens

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "data" / "asr_codeswitch"
WAV_DIR = OUT_DIR / "wav"
SENTENCES = OUT_DIR / "sentences.txt"
MANIFEST = OUT_DIR / "manifest.jsonl"
SAMPLE_RATE = 16_000


def record_one() -> np.ndarray:
    """Record until the user presses Enter again."""
    frames: queue.Queue[np.ndarray] = queue.Queue()

    def callback(indata, _frames, _time, status):
        if status:
            print(status, file=sys.stderr)
        frames.put(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback
    ):
        input("  recording... press Enter to stop")

    chunks: list[np.ndarray] = []
    while not frames.empty():
        chunks.append(frames.get())
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks, axis=0).reshape(-1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--redo", action="store_true", help="re-record sentences that already have a wav")
    args = parser.parse_args()

    if not SENTENCES.is_file():
        raise SystemExit(f"Missing {SENTENCES}.")

    WAV_DIR.mkdir(parents=True, exist_ok=True)
    sentences = [
        line.strip()
        for line in SENTENCES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    rows: list[dict] = []
    for idx, sentence in enumerate(sentences):
        wav_path = WAV_DIR / f"cs_{idx:03d}.wav"
        en_idx = english_indices(sentence)

        if wav_path.is_file() and not args.redo:
            print(f"[{idx:03d}] already recorded, skipping")
        else:
            words = tokens(sentence)
            print(f"\n[{idx:03d}] {sentence}")
            print(f"      (English words: {[words[i] for i in en_idx]})")
            input("  Enter to start recording")
            audio = record_one()
            if audio.size == 0:
                print("  nothing recorded, skipping")
                continue
            sf.write(str(wav_path), audio, SAMPLE_RATE)
            print(f"  saved {audio.size / SAMPLE_RATE:.1f}s -> {wav_path.name}")

        if wav_path.is_file():
            rows.append(
                {
                    "id": f"cs_{idx:03d}",
                    "wav": str(wav_path.relative_to(REPO_ROOT)),
                    "reference": sentence,
                    "english_indices": en_idx,
                }
            )

    MANIFEST.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    print(f"\nmanifest: {MANIFEST}  ({len(rows)} sentences)")


if __name__ == "__main__":
    main()
