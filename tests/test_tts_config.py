from __future__ import annotations

import inspect

import pytest

from soca.tts import VALTEC_TTS_CONFIG, VietnameseTTS, create_tts_engine


@pytest.fixture
def disable_tts_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(VietnameseTTS, "_ensure_loaded", lambda self: None)


def test_valtec_is_the_only_tts_configuration() -> None:
    assert VALTEC_TTS_CONFIG.key == "valtec_multispeaker"
    assert VALTEC_TTS_CONFIG.runner == "valtec"
    assert VALTEC_TTS_CONFIG.default_voice == "NF"
    assert VALTEC_TTS_CONFIG.voices == ("NF", "SF", "NM1", "SM", "NM2")


def test_factory_has_no_model_selector() -> None:
    parameters = inspect.signature(create_tts_engine).parameters

    assert set(parameters) == {"voice"}
    assert parameters["voice"].kind is inspect.Parameter.KEYWORD_ONLY


def test_factory_creates_valtec_with_default_voice(disable_tts_loader: None) -> None:
    engine = create_tts_engine()

    assert isinstance(engine, VietnameseTTS)
    assert engine.voice == "NF"


def test_factory_accepts_valtec_voice_override(disable_tts_loader: None) -> None:
    engine = create_tts_engine(voice="SF")

    assert isinstance(engine, VietnameseTTS)
    assert engine.voice == "SF"


def test_factory_rejects_positional_model_key() -> None:
    with pytest.raises(TypeError):
        create_tts_engine("other_tts")  # type: ignore[call-arg]


def test_factory_eagerly_invokes_valtec_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_ensure_loaded(self) -> None:
        calls.append(self.voice)

    monkeypatch.setattr(VietnameseTTS, "_ensure_loaded", fake_ensure_loaded)

    create_tts_engine()

    assert calls == ["NF"]
