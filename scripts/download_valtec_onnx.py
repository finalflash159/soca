from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "valtecAI-team/valtec-tts-onnx"
DEST = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "tts"
    / "valtec_multispeaker"
    / "reference"
    / "upstream"
)
REQUIRED_FILES = (
    "text_encoder.onnx",
    "duration_predictor.onnx",
    "flow.onnx",
    "decoder.onnx",
    "phoneme_dict.json",
    "precomputed_latents.json",
    "tts_config.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(DEST),
        allow_patterns=list(REQUIRED_FILES),
    )
    missing = [name for name in REQUIRED_FILES if not (DEST / name).exists()]
    if missing:
        raise FileNotFoundError(f"Valtec ONNX download incomplete: {missing}")
    (DEST / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_key": "valtec_multispeaker",
                "role": "reference",
                "artifact_id": "upstream-reference",
                "source_repo": REPO_ID,
                "active_variant": "upstream_reference",
                "variants": {
                    "upstream_reference": {
                        "precision": "upstream_fp32",
                        "runtime_graphs": {
                            "text_encoder": "text_encoder.onnx",
                            "duration_predictor": "duration_predictor.onnx",
                            "flow": "flow.onnx",
                            "decoder": "decoder.onnx",
                        },
                    }
                },
                "runtime_files": {
                    "config": "tts_config.json",
                },
                "runtime_defaults": {
                    "sample_rate": 24000,
                    "hop_length": 256,
                    "noise_scale": 0.667,
                    "length_scale": 1.0,
                    "tone_offset_vi": 16,
                    "language_id_vi": 7,
                    "add_blank": True,
                },
                "voices": {
                    "map": {"NF": 0, "SF": 1, "NM1": 2, "SM": 3, "NM2": 4},
                    "default": "NF",
                    "provenance": "upstream edge inference.py speaker_id contract",
                },
                "files": {name: _sha256(DEST / name) for name in REQUIRED_FILES},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(path)


if __name__ == "__main__":
    main()