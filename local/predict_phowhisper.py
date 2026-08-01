"""Generate PhoWhisper predictions for the code-switch set.

    uv run python local/predict_phowhisper.py --model-key phowhisper_small

Runs through RobustASR (the real production pipeline) so the measurement
reflects what is actually running, not a bare ASR call.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np

from soca.asr.robust_asr import RobustASR
from soca.asr.whisper_onnx import VietnameseASR

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "data" / "asr_codeswitch" / "manifest.jsonl"
PRED_DIR = REPO_ROOT / "data" / "asr_codeswitch" / "preds"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", default="phowhisper_small")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="bypass RobustASR and use VietnameseASR directly, to isolate guard effects",
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    inner = VietnameseASR(model_key=args.model_key, num_threads=4)
    # confidence_profile_model_key=args.model_key: the guard only enables when
    # calibration matches the runtime model, otherwise RobustASR disables it
    # and records why.
    engine = None if args.raw else RobustASR(
        asr=inner, confidence_profile_model_key=args.model_key
    )
    if engine is not None:
        print(f"confidence guard: {engine.confidence_guard_status}")

    predictions: dict[str, str] = {}
    rtfs: list[float] = []
    for row in rows:
        audio, _ = librosa.load(str(REPO_ROOT / row["wav"]), sr=16_000, mono=True)
        audio = audio.astype(np.float32)
        t0 = time.perf_counter()
        if engine is None:
            text = inner.transcribe(audio).text
        else:
            result = engine.transcribe(audio)
            text = result.text
            if not text and result.rejection_reason:
                print(f"  {row['id']}: rejected ({result.rejection_reason})")
        elapsed = time.perf_counter() - t0
        rtfs.append(elapsed / max(len(audio) / 16_000, 1e-6))
        predictions[row["id"]] = text

    system = f"{args.model_key}{'_raw' if args.raw else ''}"
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = PRED_DIR / f"{system}.json"
    out.write_text(
        json.dumps(
            {
                "system": system,
                "median_rtf": float(np.median(rtfs)),
                "predictions": predictions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}  median RTF={np.median(rtfs):.3f}")


if __name__ == "__main__":
    main()
