import pytest

from soca.asr.registry import ASR_MODEL_REGISTRY
from soca.core.profiles import (
    DEFAULT_VOICE_RUNTIME_PROFILE_KEY,
    VOICE_RUNTIME_PROFILES,
    get_voice_runtime_profile,
    validate_voice_runtime_profiles,
)
from soca.llm.registry import LLM_MODEL_REGISTRY
from soca.tts.registry import TTS_MODEL_REGISTRY


def test_default_voice_runtime_profile_exists():
    assert DEFAULT_VOICE_RUNTIME_PROFILE_KEY in VOICE_RUNTIME_PROFILES


def test_profile_keys_match_profile_key_field():
    for key, profile in VOICE_RUNTIME_PROFILES.items():
        assert profile.key == key


def test_all_profiles_reference_registered_models():
    for profile in VOICE_RUNTIME_PROFILES.values():
        assert profile.asr_model in ASR_MODEL_REGISTRY
        assert profile.llm_model in LLM_MODEL_REGISTRY
        assert profile.tts_model in TTS_MODEL_REGISTRY


def test_quality_profile_uses_omnivoice_saved_voice():
    profile = get_voice_runtime_profile("quality")

    assert profile.asr_model == "phowhisper_small"
    assert profile.llm_model == "arcee_vylinh_3b_q4_k_m"
    assert profile.tts_model == "omnivoice"
    assert profile.tts_voice == "emgai_dangiu"
    assert TTS_MODEL_REGISTRY[profile.tts_model].default_voice == "auto"


def test_edge_profile_defers_to_tts_registry_default_voice():
    profile = get_voice_runtime_profile("edge")

    assert profile.tts_model == "piper_vi_vivos_x_low"
    assert profile.tts_voice is None


def test_balanced_vieneu_profile_uses_turbo_challenger():
    profile = get_voice_runtime_profile("balanced_vieneu")

    assert profile.asr_model == "phowhisper_base"
    assert profile.llm_model == "arcee_vylinh_3b_q4_k_m"
    assert profile.tts_model == "vieneu_v2_turbo"
    assert profile.tts_voice == "XuanVinh"


def test_unknown_profile_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown voice runtime profile"):
        get_voice_runtime_profile("not_real")


def test_runtime_profile_validation_passes():
    assert validate_voice_runtime_profiles() == []
