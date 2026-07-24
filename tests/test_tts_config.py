from __future__ import annotations

import inspect

import pytest

import soca.tts.factory as factory
from soca.tts import VALTEC_TTS_CONFIG, TTSRuntimeUnavailableError, create_tts_engine


def test_valtec_is_the_only_tts_configuration() -> None:
    assert VALTEC_TTS_CONFIG.key == "valtec_multispeaker"
    assert VALTEC_TTS_CONFIG.runner == "valtec_onnx"
    assert VALTEC_TTS_CONFIG.default_voice == "NF"
    assert VALTEC_TTS_CONFIG.voices == ("NF", "SF", "NM1", "SM", "NM2")


def test_factory_has_no_model_selector() -> None:
    parameters = inspect.signature(create_tts_engine).parameters

    assert set(parameters) == {"voice"}
    assert parameters["voice"].kind is inspect.Parameter.KEYWORD_ONLY


def test_factory_rejects_positional_model_key() -> None:
    with pytest.raises(TypeError):
        create_tts_engine("other_tts")  # type: ignore[call-arg]


def test_valtec_config_points_to_onnx() -> None:
    config = VALTEC_TTS_CONFIG

    assert config.runner == "valtec_onnx"
    assert config.default_voice == "NF"


def test_factory_wraps_missing_artifact_in_runtime_unavailable(monkeypatch) -> None:
    def _missing() -> object:
        raise FileNotFoundError("no current.json")

    monkeypatch.setattr(factory, "resolve_current_valtec_release", _missing)

    with pytest.raises(TTSRuntimeUnavailableError):
        create_tts_engine()
