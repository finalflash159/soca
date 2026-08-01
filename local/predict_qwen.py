"""Generate Qwen3-ASR predictions for the code-switch set, with and without context.

MUST run with .venv-qwen (see zplan/qwen3_asr_probe_plan.vi.md §Q0):
    .venv-qwen/bin/python local/predict_qwen.py --context none
    .venv-qwen/bin/python local/predict_qwen.py --context tech
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "data" / "asr_codeswitch" / "manifest.jsonl"
PRED_DIR = REPO_ROOT / "data" / "asr_codeswitch" / "preds"

MODEL_ID = "Qwen/Qwen3-ASR-0.6B"

CONTEXTS = {
    "none": "",
    "tech": (
        "Cuộc hội thoại về lập trình. Giữ nguyên cách viết các thuật ngữ tiếng Anh: "
        "GitHub, PyTorch, TensorFlow, TypeScript, PostgreSQL, Docker, Kubernetes, "
        "ONNX Runtime, Redis, Kafka, Nginx, FastAPI, Flask, FAISS, llama.cpp, "
        "API, JSON, GGUF, embedding, transformer, inference, latency, quantize."
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", default="tech", choices=sorted(CONTEXTS))
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument(
        "--language",
        default="Vietnamese",
        help="'auto' lets the model detect language; forcing Vietnamese is more stable",
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    args = parser.parse_args()

    from qwen_asr import Qwen3ASRModel

    rows = [
        json.loads(line)
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    model = Qwen3ASRModel.from_pretrained(
        args.model_id,
        dtype=torch.float32 if args.dtype == "float32" else torch.bfloat16,
        device_map=args.device,
        max_new_tokens=256,
    )
    context = CONTEXTS[args.context]
    language = None if args.language == "auto" else args.language

    import soundfile as sf

    predictions: dict[str, str] = {}
    rtfs: list[float] = []
    for row in rows:
        wav = REPO_ROOT / row["wav"]
        info = sf.info(str(wav))
        audio_sec = info.frames / info.samplerate
        t0 = time.perf_counter()
        result = model.transcribe(audio=str(wav), context=context, language=language)[0]
        rtfs.append((time.perf_counter() - t0) / max(audio_sec, 1e-6))
        predictions[row["id"]] = result.text

    model_size = "1.7b" if "1.7B" in args.model_id else "0.6b"
    system = f"qwen3_asr_{model_size}_ctx_{args.context}"
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    out = PRED_DIR / f"{system}.json"
    out.write_text(
        json.dumps(
            {
                "system": system,
                "median_rtf": float(np.median(rtfs)),
                "device": args.device,
                "dtype": args.dtype,
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
