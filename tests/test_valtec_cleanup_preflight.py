import json

import pytest

from soca.tts.valtec import resolve_current_valtec_release, resolve_valtec_onnx_artifacts


@pytest.mark.real_model
def test_active_valtec_release_is_rollbackable():
    current = resolve_current_valtec_release()
    artifacts = resolve_valtec_onnx_artifacts(current, verify_checksums=True)
    manifest = json.loads((current / "manifest.json").read_text(encoding="utf-8"))
    assert artifacts.role == "release"
    assert manifest["acceptance"]["gates"]["onnx_smoke"] is True
    assert manifest["acceptance"]["gates"]["voice_listening"] is True
