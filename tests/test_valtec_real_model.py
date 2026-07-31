from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from soca.tts.valtec import (
    ValtecOnnxTTS,
    ValtecVietnameseFrontend,
    resolve_valtec_onnx_artifacts,
)
from soca.tts.valtec.foreign_g2p import ChainedForeignG2P

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


@pytest.mark.real_model
def test_foreign_g2p_opt_in_renders_oov_english_through_the_real_pipeline(
    tmp_path: Path,
) -> None:
    """End-to-end check for the config.get("foreign_g2p") == "g2p_en" seam.

    Unit tests stub the backend; this exercises the exact wiring
    ValtecVietnameseFrontend.from_artifacts() builds in production, against
    the four real ONNX sessions, so a regression in the wiring itself (not
    just the backend logic) would show up here.
    """
    pytest.importorskip("g2p_en")
    if not (REFERENCE_ROOT / "manifest.json").is_file():
        pytest.skip("download the upstream Valtec ONNX reference first")

    staged_root = tmp_path / "upstream"
    shutil.copytree(REFERENCE_ROOT, staged_root)
    config_path = staged_root / "tts_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["foreign_g2p"] = "g2p_en"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    artifacts = resolve_valtec_onnx_artifacts(staged_root, allow_reference=True)
    frontend = ValtecVietnameseFrontend.from_artifacts(artifacts)
    assert isinstance(frontend.g2p.foreign_g2p, ChainedForeignG2P)

    engine = ValtecOnnxTTS(
        artifact_root=staged_root,
        artifact_variant="upstream_reference",
        allow_reference=True,
        frontend=frontend,
    )

    # "github" is OOV for the CMU path and was measured mispronounced by
    # g2p_en alone (plan §6.1c); the shipped lexicon must still catch it.
    result = engine.synthesize("Xem thử pytorch trên github", voice="NF")

    assert result.audio.size > 0
    assert engine.frontend_metadata["backend"] == "portable_web_port"
    assert engine.frontend_metadata["unknown_phoneme_count"] == 0
