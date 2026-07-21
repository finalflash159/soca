from __future__ import annotations

from dataclasses import dataclass

from soca.asr.registry import ASR_MODEL_REGISTRY
from soca.llm.registry import LLM_MODEL_REGISTRY
from soca.tts.registry import TTS_MODEL_REGISTRY


@dataclass(frozen=True)
class VoiceRuntimeProfile:
    key: str
    description: str
    asr_model: str
    llm_model: str
    tts_model: str
    tts_voice: str | None = None
    endpoint_silence_ms: int = 700
    adaptive_endpoint: bool = True
    max_record_ms: int = 10000
    max_tokens: int = 160
    temperature: float = 0.2
    top_p: float = 0.95


DEFAULT_VOICE_RUNTIME_PROFILE_KEY = "baseline"

VOICE_RUNTIME_PROFILES: dict[str, VoiceRuntimeProfile] = {
    "baseline": VoiceRuntimeProfile(
        key="baseline",
        description="Known local baseline profile for repeatable voice-loop testing.",
        asr_model="phowhisper_base",
        llm_model="arcee_vylinh_3b_q4_k_m",
        tts_model="valtec_multispeaker",
        tts_voice="NF",
    ),
    "quality": VoiceRuntimeProfile(
        key="quality",
        description="High-quality candidate profile using the saved OmniVoice voice artifact.",
        asr_model="phowhisper_small",
        llm_model="arcee_vylinh_3b_q4_k_m",
        tts_model="omnivoice",
        tts_voice="emgai_dangiu",
    ),
    "edge": VoiceRuntimeProfile(
        key="edge",
        description="Edge-oriented candidate profile for lightweight fallback testing.",
        asr_model="phowhisper_base",
        llm_model="qwen3_0_6b_q8_0",
        tts_model="piper_vi_vivos_x_low",
        tts_voice=None,
    ),
    "balanced_vieneu": VoiceRuntimeProfile(
        key="balanced_vieneu",
        description="Practical quality/latency challenger using VieNeu Turbo.",
        asr_model="phowhisper_base",
        llm_model="arcee_vylinh_3b_q4_k_m",
        tts_model="vieneu_v2_turbo",
        tts_voice="XuanVinh",
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

    for key, profile in VOICE_RUNTIME_PROFILES.items():
        if profile.key != key:
            errors.append(f"{key}: profile.key must match dict key")

        if profile.asr_model not in ASR_MODEL_REGISTRY:
            errors.append(f"{key}: unknown ASR model {profile.asr_model!r}")

        if profile.llm_model not in LLM_MODEL_REGISTRY:
            errors.append(f"{key}: unknown LLM model {profile.llm_model!r}")

        if profile.tts_model not in TTS_MODEL_REGISTRY:
            errors.append(f"{key}: unknown TTS model {profile.tts_model!r}")

        if profile.tts_voice is not None and not profile.tts_voice.strip():
            errors.append(f"{key}: tts_voice must be None or a non-empty string")

    return errors
