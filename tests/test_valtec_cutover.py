from __future__ import annotations

import soca.tts.factory as factory
from soca.tts import VALTEC_TTS_CONFIG


class FakeArtifacts:
    artifact_id = "release-1"
    variant = "fp32"
    precision = "fp32"
    role = "release"


class FakeFrontend:
    @classmethod
    def from_artifacts(cls, artifacts):
        assert artifacts is FakeArtifacts
        return cls()


class FakeEngine:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_factory_builds_frontend_and_runner_from_same_active_release(monkeypatch, tmp_path):
    release = tmp_path / "releases/release-1"
    monkeypatch.setattr(factory, "resolve_current_valtec_release", lambda: release)
    monkeypatch.setattr(factory, "resolve_valtec_onnx_artifacts", lambda root: FakeArtifacts)
    monkeypatch.setattr(factory, "ValtecVietnameseFrontend", FakeFrontend)
    monkeypatch.setattr(factory, "ValtecOnnxTTS", FakeEngine)
    engine = factory.create_tts_engine(voice="SF")
    assert engine.kwargs["artifact_root"] == release
    assert isinstance(engine.kwargs["frontend"], FakeFrontend)
    assert engine.kwargs["voice"] == "SF"
    assert engine.kwargs["length_scale"] == VALTEC_TTS_CONFIG.length_scale
    assert "config" not in engine.kwargs


def test_stable_config_does_not_advertise_upstream_reference_as_runtime():
    config = VALTEC_TTS_CONFIG
    assert config.default_voice == "NF"
    assert config.runner == "valtec_onnx"
