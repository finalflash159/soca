from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from soca.tts.valtec.artifacts import (
    activate_valtec_release,
    resolve_current_valtec_release,
    resolve_valtec_onnx_artifacts,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifact(root: Path, *, role: str = "release") -> Path:
    root.mkdir(parents=True)
    paths = {
        "fp32/text_encoder.onnx": b"encoder",
        "fp32/duration_predictor.onnx": b"duration",
        "fp32/flow.onnx": b"flow",
        "fp32/decoder.onnx": b"decoder",
        "tts_config.json": b"{}",
        "phoneme_dict.json": b"{}",
        "precomputed_latents.json": b"{}",
    }
    for relative, content in paths.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    manifest = {
        "schema_version": 1,
        "model_key": "valtec_multispeaker",
        "artifact_id": root.name,
        "role": role,
        "active_variant": "fp32",
        "variants": {
            "fp32": {
                "precision": "fp32",
                "release_eligible": True,
                "runtime_graphs": {
                    "text_encoder": "fp32/text_encoder.onnx",
                    "duration_predictor": "fp32/duration_predictor.onnx",
                    "flow": "fp32/flow.onnx",
                    "decoder": "fp32/decoder.onnx",
                },
            }
        },
        "runtime_files": {
            "config": "tts_config.json",
        },
        "runtime_defaults": {
            "sample_rate": 24000,
            "hop_length": 512,
            "noise_scale": 0.667,
            "length_scale": 1.0,
            "tone_offset_vi": 16,
            "language_id_vi": 7,
            "add_blank": True,
        },
        "voices": {
            "map": {"NF": 0, "SF": 1, "NM1": 2, "SM": 3, "NM2": 4},
            "default": "NF",
        },
        "files": {relative: _sha256(root / relative) for relative in paths},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_resolver_returns_validated_runtime_contract(tmp_path):
    root = _write_artifact(tmp_path / "release-1")
    artifact = resolve_valtec_onnx_artifacts(root, verify_checksums=True)
    assert artifact.variant == "fp32"
    assert artifact.sample_rate == 24000
    assert artifact.voice_map["NF"] == 0
    assert artifact.decoder == root / "fp32/decoder.onnx"


def test_reference_requires_explicit_opt_in(tmp_path):
    root = _write_artifact(tmp_path / "upstream-reference", role="reference")
    with pytest.raises(ValueError, match="not allowed"):
        resolve_valtec_onnx_artifacts(root)
    assert resolve_valtec_onnx_artifacts(root, allow_reference=True).role == "reference"


def test_resolver_rejects_path_traversal(tmp_path):
    root = _write_artifact(tmp_path / "release-1")
    path = root / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["variants"]["fp32"]["runtime_graphs"]["decoder"] = "../decoder.onnx"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsafe Valtec manifest path"):
        resolve_valtec_onnx_artifacts(root)


def test_activation_is_atomic_and_current_pointer_resolves(tmp_path):
    model_root = tmp_path / "valtec_multispeaker"
    release = _write_artifact(model_root / "releases" / "release-1")
    activate_valtec_release("release-1", model_root)
    pointer = json.loads((model_root / "current.json").read_text(encoding="utf-8"))
    assert pointer["artifact_id"] == "release-1"
    assert pointer["manifest_sha256"] == _sha256(release / "manifest.json")
    assert resolve_current_valtec_release(model_root) == release.resolve()


def test_activation_rejects_corrupted_runtime_file(tmp_path):
    model_root = tmp_path / "valtec_multispeaker"
    release = _write_artifact(model_root / "releases" / "release-1")
    (release / "fp32/decoder.onnx").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch: fp32/decoder.onnx"):
        activate_valtec_release("release-1", model_root)


def test_active_variant_must_be_release_eligible(tmp_path):
    root = _write_artifact(tmp_path / "release-1")
    path = root / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["variants"]["fp32"]["release_eligible"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="has not passed release gates"):
        resolve_valtec_onnx_artifacts(root)
