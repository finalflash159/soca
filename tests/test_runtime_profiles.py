import pytest

from soca.asr.qwen_artifacts import QWEN_ARTIFACT_REGISTRY
from soca.asr.registry import ASR_MODEL_REGISTRY
from soca.asr.selection import ASREngine, ASRSelection
from soca.core.profiles import (
    DEFAULT_VOICE_RUNTIME_PROFILE_KEY,
    VOICE_RUNTIME_PROFILES,
    VoiceRuntimeProfile,
    get_voice_runtime_profile,
    validate_voice_runtime_profiles,
)
from soca.llm.registry import LLM_MODEL_REGISTRY
from soca.tts import VALTEC_TTS_CONFIG


def test_default_voice_runtime_profile_is_baseline() -> None:
    assert DEFAULT_VOICE_RUNTIME_PROFILE_KEY == "baseline"
    assert DEFAULT_VOICE_RUNTIME_PROFILE_KEY in VOICE_RUNTIME_PROFILES


def test_profile_keys_match_profile_key_field() -> None:
    assert all(key == profile.key for key, profile in VOICE_RUNTIME_PROFILES.items())


def test_all_profiles_reference_registered_models() -> None:
    for profile in VOICE_RUNTIME_PROFILES.values():
        registry = (
            ASR_MODEL_REGISTRY
            if profile.asr.engine is ASREngine.PHOWHISPER
            else QWEN_ARTIFACT_REGISTRY
        )
        assert profile.asr_model in registry
        assert profile.llm_model in LLM_MODEL_REGISTRY


def test_qwen_profiles_are_explicit_and_baseline_remains_default() -> None:
    assert set(VOICE_RUNTIME_PROFILES) == {
        "baseline",
        "qwen-reference",
        "qwen-release",
    }
    assert {profile.tts_voice for profile in VOICE_RUNTIME_PROFILES.values()} == {
        VALTEC_TTS_CONFIG.default_voice
    }
    assert set(VALTEC_TTS_CONFIG.voices) == {"NF", "SF", "NM1", "SM", "NM2"}


def test_baseline_uses_the_former_quality_stack() -> None:
    baseline = get_voice_runtime_profile("baseline")

    assert baseline.asr_model == "phowhisper_small"
    assert baseline.llm_model == "arcee_vylinh_3b_q4_k_m"
    assert not hasattr(baseline, "tts_model")
    assert baseline.tts_voice == VALTEC_TTS_CONFIG.default_voice


@pytest.mark.parametrize("profile_key", ["quality", "edge", "balanced_vieneu"])
def test_removed_profiles_are_not_exposed(profile_key: str) -> None:
    assert profile_key not in VOICE_RUNTIME_PROFILES
    with pytest.raises(ValueError, match="Unknown voice runtime profile"):
        get_voice_runtime_profile(profile_key)


def test_unknown_profile_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown voice runtime profile"):
        get_voice_runtime_profile("not_real")


def test_runtime_profile_validation_passes() -> None:
    assert validate_voice_runtime_profiles() == []


def test_runtime_profile_validation_rejects_an_unknown_extra_profile(monkeypatch) -> None:
    monkeypatch.setitem(
        VOICE_RUNTIME_PROFILES,
        "quality",
        VoiceRuntimeProfile(
            key="quality",
            description="Invalid duplicate product profile.",
            asr=ASRSelection.phowhisper("phowhisper_small"),
            llm_model="arcee_vylinh_3b_q4_k_m",
            tts_voice=VALTEC_TTS_CONFIG.default_voice,
        ),
    )

    error = validate_voice_runtime_profiles()[0]
    assert "runtime profiles must contain exactly" in error
    assert "quality" in error
