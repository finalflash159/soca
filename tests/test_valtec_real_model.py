from __future__ import annotations

from pathlib import Path

import pytest

from soca.tts.valtec import (
    ValtecOnnxTTS,
    ValtecVietnameseFrontend,
    resolve_valtec_onnx_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = (
    REPO_ROOT / "models/tts/valtec_multispeaker/reference/upstream"
)


def _require_upstream_reference():
    if not (REFERENCE_ROOT / "manifest.json").is_file():
        pytest.skip("download the upstream Valtec ONNX reference first")
    return resolve_valtec_onnx_artifacts(
        REFERENCE_ROOT,
        allow_reference=True,
        verify_checksums=True,
    )


@pytest.mark.real_model
def test_upstream_reference_real_audio_smoke() -> None:
    artifacts = _require_upstream_reference()
    frontend = ValtecVietnameseFrontend.from_artifacts(artifacts)
    engine = ValtecOnnxTTS(
        artifact_root=REFERENCE_ROOT,
        artifact_variant="upstream_reference",
        allow_reference=True,
        frontend=frontend,
    )

    result = engine.synthesize(
        "Họp lúc 14:30 ngày 15/08/1990",
        voice="NF",
    )

    assert artifacts.role == "reference"
    assert result.audio.size > 0
    assert result.sample_rate == artifacts.sample_rate == 24_000
    assert engine.frontend_metadata == {
        "backend": "portable_web_port",
        "unknown_phoneme_count": 0,
    }