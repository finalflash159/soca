from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path

import soundfile as sf

from soca.asr.qwen_backend import QwenASRBackend


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifact-key", required=True)
    parser.add_argument("--artifact-revision", required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument(
        "--run-type",
        choices=("release_benchmark", "research_benchmark"),
        required=True,
    )
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-dirty", action="store_true")
    parser.add_argument("--raw-log-reference", required=True)
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
                "schema_version": 1,
                "run_type": args.run_type,
                "layout": args.layout,
                "source": {
                    "revision": args.source_revision,
                    "dirty": args.source_dirty,
                },
                "artifact": {
                    "key": args.artifact_key,
                    "revision": args.artifact_revision,
                    "local_snapshot": str(args.model.resolve()),
                },
                "configuration": {
                    "context": "",
                    "device": "cpu",
                    "dtype": "float32",
                    "language": "Vietnamese",
                    "max_new_tokens": 128,
                    "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
                    "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE") == "1",
                    "lock_sha256": _sha256(args.lock),
                },
                "fixtures": [
                    {"name": path.name, "sha256": _sha256(path)} for path in args.audio
                ],
                "python": platform.python_version(),
                "platform": platform.platform(),
                "startup_s": startup_s,
                "max_rss_bytes": _max_rss_bytes(),
                "records": records,
                "failures": [],
                "raw_log_reference": args.raw_log_reference,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
