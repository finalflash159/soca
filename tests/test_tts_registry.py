from __future__ import annotations

import json

import numpy as np
import pytest

from soca.tts import (
    DEFAULT_TTS_MODEL_KEY,
    TIER_A_TTS_MODEL_KEYS,
    TIER_B_TTS_MODEL_KEYS,
    TTS_MODEL_REGISTRY,
    ExternalCommandTTS,
    F5VietnameseTTS,
    KaniTTSRunner,
    OmniVoiceTTS,
    VieneuTTS,
    VietnameseTTS,
    VietTTSServerTTS,
    create_tts_engine,
    get_tts_model_config,
)
from soca.tts.mms_runner import MMSTTS
from soca.tts.piper_runner import PiperTTS


@pytest.fixture
def disable_tts_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    for cls in (
        VietnameseTTS,
        MMSTTS,
        PiperTTS,
        VieneuTTS,
        KaniTTSRunner,
        VietTTSServerTTS,
        ExternalCommandTTS,
        F5VietnameseTTS,
        OmniVoiceTTS,
    ):
        monkeypatch.setattr(cls, "_ensure_loaded", lambda self: None)


def test_default_tts_model_key_exists() -> None:
    assert DEFAULT_TTS_MODEL_KEY in TTS_MODEL_REGISTRY


def test_tier_a_keys_exist_in_registry() -> None:
    for model_key in TIER_A_TTS_MODEL_KEYS:
        assert model_key in TTS_MODEL_REGISTRY


def test_tier_b_keys_exist_in_registry() -> None:
    for model_key in TIER_B_TTS_MODEL_KEYS:
        assert model_key in TTS_MODEL_REGISTRY


def test_registry_keys_match_config_keys() -> None:
    for model_key, config in TTS_MODEL_REGISTRY.items():
        assert config.key == model_key


def test_get_tts_model_config_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="Unknown TTS model key"):
        get_tts_model_config("not_real")


def test_factory_creates_valtec(disable_tts_loaders) -> None:
    engine = create_tts_engine("valtec_multispeaker")

    assert isinstance(engine, VietnameseTTS)
    assert engine.voice == "NF"


def test_factory_creates_mms(disable_tts_loaders) -> None:
    engine = create_tts_engine("mms_tts_vie")

    assert isinstance(engine, MMSTTS)


def test_factory_creates_piper(disable_tts_loaders) -> None:
    engine = create_tts_engine("piper_vi_vivos_x_low")

    assert isinstance(engine, PiperTTS)


@pytest.mark.parametrize(
    ("model_key", "expected_type"),
    [
        ("vieneu_v2_turbo", VieneuTTS),
        ("vieneu_v2_standard", VieneuTTS),
        ("kani_370m_vie", KaniTTSRunner),
        ("viet_tts_onnx", VietTTSServerTTS),
        ("vixtts", ExternalCommandTTS),
        ("f5_vi_hynt", F5VietnameseTTS),
        ("f5_vi_zalopay", F5VietnameseTTS),
        ("omnivoice", OmniVoiceTTS),
    ],
)
def test_factory_creates_candidate_runners(
    model_key: str,
    expected_type: type,
    disable_tts_loaders,
) -> None:
    engine = create_tts_engine(model_key)

    assert isinstance(engine, expected_type)


def test_factory_eagerly_invokes_runner_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_ensure_loaded(self) -> None:
        calls.append(self.voice)

    monkeypatch.setattr(VietnameseTTS, "_ensure_loaded", fake_ensure_loaded)

    create_tts_engine("valtec_multispeaker")

    assert calls == ["NF"]


def test_piper_wav_bytes_to_float32_mono() -> None:
    import io
    import wave

    payload = io.BytesIO()
    samples = (np.ones(160, dtype=np.float32) * 0.25 * 32767).astype(np.int16)

    with wave.open(payload, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(samples.tobytes())

    audio, sample_rate = PiperTTS._wav_bytes_to_float32_mono(payload.getvalue())

    assert sample_rate == 16000
    assert audio.dtype == np.float32
    assert audio.shape == (160,)


def test_omnivoice_supports_json_voice_design_aliases(
    monkeypatch: pytest.MonkeyPatch,
    disable_tts_loaders,
) -> None:
    config = get_tts_model_config("omnivoice")
    monkeypatch.setenv(
        config.voices_env_var,
        '{"female_young": "female, young adult, moderate pitch", "male_low": "male, low pitch"}',
    )

    engine = OmniVoiceTTS(config)

    assert "auto" in engine.list_voices()
    assert "female_young" in engine.list_voices()
    assert "male_low" in engine.list_voices()
    assert engine._instruction_for_voice("auto") is None
    assert engine._instruction_for_voice("female_young") == "female, young adult, moderate pitch"


def test_omnivoice_exposes_registry_voice_designs_by_default(
    monkeypatch: pytest.MonkeyPatch,
    disable_tts_loaders,
) -> None:
    config = get_tts_model_config("omnivoice")
    monkeypatch.delenv(config.voices_env_var, raising=False)

    engine = OmniVoiceTTS(config)

    assert engine.list_voices()[:5] == [
        "auto",
        "female_young",
        "female_high",
        "male_low",
        "male_deep",
    ]
    assert engine._instruction_for_voice("male_low") == "male, middle-aged, low pitch"
    assert config.smoke_test_voices == ("female_young", "female_high", "male_low", "male_deep")


def test_omnivoice_supports_semicolon_voice_design_aliases(
    monkeypatch: pytest.MonkeyPatch,
    disable_tts_loaders,
) -> None:
    config = get_tts_model_config("omnivoice")
    monkeypatch.setenv(
        config.voices_env_var,
        "female_young=female, young adult, moderate pitch;male_low=male, low pitch",
    )

    engine = OmniVoiceTTS(config)

    assert "auto" in engine.list_voices()
    assert "female_young" in engine.list_voices()
    assert "male_low" in engine.list_voices()
    assert engine._instruction_for_voice("male_low") == "male, low pitch"


def test_omnivoice_lists_saved_voice_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    disable_tts_loaders,
) -> None:
    import soca.tts.registry as registry

    monkeypatch.setattr(registry, "TTS_MODELS_ROOT", tmp_path)
    config = registry.TTSModelConfig(
        key="omnivoice_test",
        display_name="OmniVoice test",
        role="test",
        runner="omnivoice",
        default_voice="auto",
        license="test",
        source_url="test",
        notes="test",
        voice_designs={"female_young": "female, young adult, moderate pitch"},
    )
    voice_dir = config.local_dir / "voices" / "sonca_female_01"
    voice_dir.mkdir(parents=True)
    (voice_dir / "prompt.pt").write_bytes(b"placeholder")
    (voice_dir / "meta.json").write_text(json.dumps({"voice_id": "sonca_female_01"}), encoding="utf-8")

    engine = OmniVoiceTTS(config)

    assert engine.list_voices() == ["auto", "female_young", "sonca_female_01"]
