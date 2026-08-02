from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import soundfile as sf

from soca.asr.qwen_backend import QwenASRBackend


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("audio", nargs="+", type=Path)
    args = parser.parse_args()

    started = time.perf_counter()
    backend = QwenASRBackend(model_id=str(args.model), max_new_tokens=128)
    startup_s = time.perf_counter() - started
    records: list[dict[str, object]] = []
    for audio_path in args.audio:
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        if sample_rate != 16_000:
            raise ValueError(f"{audio_path}: expected 16000 Hz, got {sample_rate}")
        result = backend.transcribe(audio, max_new_tokens=128, context="")
        records.append(
            {
                "audio": audio_path.name,
                "text": result.text,
                "avg_logprob": result.avg_logprob,
                "avg_logprob_reliable": result.avg_logprob_reliable,
                "latency_ms": result.latency_ms,
                "rtf": result.rtf,
            }
        )
    print(
        json.dumps(
            {
                "layout": args.layout,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "startup_s": startup_s,
                "max_rss_bytes": _max_rss_bytes(),
                "records": records,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
