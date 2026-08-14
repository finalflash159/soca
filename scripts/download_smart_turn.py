"""Download and verify the pinned Smart Turn v3.2 production artifact."""
from __future__ import annotations

import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "pipecat-ai/smart-turn-v3"
MODEL_REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
MODEL_FILE = "smart-turn-v3.2-cpu.onnx"  # CPU int8
MODEL_SHA256 = "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f"
DEST = Path(__file__).resolve().parents[1] / "models" / "smart-turn-v3-onnx"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            revision=MODEL_REVISION,
            filename=MODEL_FILE,
            local_dir=DEST,
        )
    )
    actual = _sha256(downloaded)
    if actual != MODEL_SHA256:
        raise RuntimeError(
            f"Smart Turn SHA-256 mismatch: expected {MODEL_SHA256}, got {actual}"
        )
    print("->", downloaded)


if __name__ == "__main__":
    main()
