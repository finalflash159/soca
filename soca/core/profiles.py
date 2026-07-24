from __future__ import annotations

from dataclasses import dataclass

from soca.asr.registry import ASR_MODEL_REGISTRY
from soca.llm.registry import LLM_MODEL_REGISTRY
from soca.tts.config import VALTEC_TTS_CONFIG


@dataclass(frozen=True)
class VoiceRuntimeProfile:
    key: str
    description: str
    asr_model: str
    llm_model: str
    tts_voice: str | None = None
    endpoint_silence_ms: int = 700
    adaptive_endpoint: bool = True
    max_record_ms: int = 10000
    max_tokens: int = 160
    temperature: float = 0.2
    top_p: float = 0.95
    first_clause_enabled: bool = True
    first_clause_min_chars: int = 12
    first_clause_min_words: int = 2
    first_clause_max_scan_chars: int = 80
    pcm_crossfade_enabled: bool = True
    pcm_crossfade_ms: float = 12.0


DEFAULT_VOICE_RUNTIME_PROFILE_KEY = "baseline"

VOICE_RUNTIME_PROFILES: dict[str, VoiceRuntimeProfile] = {
    "baseline": VoiceRuntimeProfile(
        key="baseline",
        description="Default high-accuracy local voice runtime using Valtec TTS.",
        asr_model="phowhisper_small",
        llm_model="arcee_vylinh_3b_q4_k_m",
        tts_voice=VALTEC_TTS_CONFIG.default_voice,
    ),
}


def get_voice_runtime_profile(profile_key: str) -> VoiceRuntimeProfile:
    try:
        return VOICE_RUNTIME_PROFILES[profile_key]
    except KeyError as exc:
        valid = ", ".join(sorted(VOICE_RUNTIME_PROFILES))
        raise ValueError(
            f"Unknown voice runtime profile: {profile_key}. Valid profiles: {valid}"
        ) from exc


def validate_voice_runtime_profiles() -> list[str]:
    errors: list[str] = []
    expected_keys = {DEFAULT_VOICE_RUNTIME_PROFILE_KEY}
    actual_keys = set(VOICE_RUNTIME_PROFILES)
    if actual_keys != expected_keys:
        errors.append(
            "runtime profiles must contain exactly "
            f"{sorted(expected_keys)!r}, got {sorted(actual_keys)!r}"
        )

    for key, profile in VOICE_RUNTIME_PROFILES.items():
        if profile.key != key:
            errors.append(f"{key}: profile.key must match dict key")
        if profile.asr_model not in ASR_MODEL_REGISTRY:
            errors.append(f"{key}: unknown ASR model {profile.asr_model!r}")
        if profile.llm_model not in LLM_MODEL_REGISTRY:
            errors.append(f"{key}: unknown LLM model {profile.llm_model!r}")
        if profile.tts_voice not in VALTEC_TTS_CONFIG.voices:
            valid = ", ".join(VALTEC_TTS_CONFIG.voices)
            errors.append(f"{key}: unknown Valtec voice {profile.tts_voice!r}; valid: {valid}")

    return errors
