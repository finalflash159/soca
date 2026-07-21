"""Download Smart Turn v3.1 ONNX weights (once) into models/ for offline use."""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "pipecat-ai/smart-turn-v3"
MODEL_FILE = "smart-turn-v3.2-cpu.onnx"      # CPU int8
DEST = Path(__file__).resolve().parents[1] / "models" / "smart-turn-v3-onnx"

def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    print("->", hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE, local_dir=DEST))

if __name__ == "__main__":
    main()
