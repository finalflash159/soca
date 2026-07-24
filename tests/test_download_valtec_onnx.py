from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download_valtec_onnx.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("download_valtec_onnx", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_writes_reference_manifest_with_checksums(tmp_path, monkeypatch):
    module = _load_module()
    destination = tmp_path / "reference"
    monkeypatch.setattr(module, "DEST", destination)

    def fake_snapshot_download(*, repo_id, local_dir, allow_patterns):
        assert repo_id == module.REPO_ID
        assert Path(local_dir) == destination
        assert tuple(allow_patterns) == module.REQUIRED_FILES
        destination.mkdir(parents=True, exist_ok=True)
        for index, name in enumerate(module.REQUIRED_FILES):
            (destination / name).write_bytes(f"fixture-{index}".encode())
        return str(destination)

    monkeypatch.setattr(module, "snapshot_download", fake_snapshot_download)
    module.main()

    payload = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["model_key"] == "valtec_multispeaker"
    assert payload["role"] == "reference"
    assert payload["active_variant"] == "upstream_reference"
    assert payload["voices"]["map"] == {"NF": 0, "SF": 1, "NM1": 2, "SM": 3, "NM2": 4}
    assert payload["voices"]["default"] == "NF"
    assert payload["runtime_defaults"]["sample_rate"] == 24000
    assert set(payload["files"]) == set(module.REQUIRED_FILES)
    for name in module.REQUIRED_FILES:
        expected = hashlib.sha256((destination / name).read_bytes()).hexdigest()
        assert payload["files"][name] == expected


def test_download_fails_before_manifest_when_required_file_is_missing(tmp_path, monkeypatch):
    module = _load_module()
    destination = tmp_path / "reference"
    monkeypatch.setattr(module, "DEST", destination)

    def incomplete_download(**_kwargs):
        destination.mkdir(parents=True, exist_ok=True)
        for name in module.REQUIRED_FILES[:-1]:
            (destination / name).write_bytes(b"fixture")
        return str(destination)

    monkeypatch.setattr(module, "snapshot_download", incomplete_download)

    with pytest.raises(FileNotFoundError, match="tts_config.json"):
        module.main()
    assert not (destination / "manifest.json").exists()


def test_required_files_are_an_explicit_allow_list():
    module = _load_module()
    assert module.REQUIRED_FILES == (
        "text_encoder.onnx",
        "duration_predictor.onnx",
        "flow.onnx",
        "decoder.onnx",
        "phoneme_dict.json",
        "precomputed_latents.json",
        "tts_config.json",
    )
