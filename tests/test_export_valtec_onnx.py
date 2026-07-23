from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import onnx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/export_valtec_onnx.py"
SOURCE_ROOT = REPO_ROOT / "external/valtec-tts"


def test_export_script_does_not_mutate_sys_path():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "sys.path" not in source
    assert "from infer" not in source


@pytest.mark.real_model
def test_real_checkpoint_exports_four_dynamic_graphs(tmp_path):
    checkpoint_value = os.environ.get("SOCA_VALTEC_TEST_CHECKPOINT")
    config_value = os.environ.get("SOCA_VALTEC_TEST_CONFIG")
    if not checkpoint_value or not config_value:
        pytest.skip("Set SOCA_VALTEC_TEST_CHECKPOINT and SOCA_VALTEC_TEST_CONFIG")
    # The subprocess runs with cwd=SOURCE_ROOT so the vendored `src` package is
    # importable, which means relative --checkpoint/--config paths would resolve
    # against external/valtec-tts. Resolve them against the caller's cwd first so
    # the plan's relative-path invocation still works.
    checkpoint_path = str(Path(checkpoint_value).resolve())
    config_path = str(Path(config_value).resolve())
    output = tmp_path / "fp32"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE_ROOT)
    subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--checkpoint", checkpoint_path,
            "--config", config_path,
            "--output-dir", str(output),
            "--trust-checkpoint",
        ],
        cwd=SOURCE_ROOT,
        env=environment,
        check=True,
    )
    for name in ("text_encoder", "duration_predictor", "flow", "decoder"):
        graph = onnx.load(output / f"{name}.onnx")
        onnx.checker.check_model(graph, full_check=True)
    runtime = json.loads((tmp_path / "tts_config.json").read_text(encoding="utf-8"))
    assert runtime["speaker_id_map"].keys() == {"NF", "SF", "NM1", "SM", "NM2"}
    assert runtime["language_id_map"]["VI"] == 7
    assert runtime["tone_offset_vi"] == 16