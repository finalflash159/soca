from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TTS_MODELS_ROOT = REPO_ROOT / "models" / "tts"


@dataclass(frozen=True)
class TTSModelConfig:
    key: str
    display_name: str
    role: str
    runner: str
    default_voice: str
    license: str
    source_url: str
    notes: str
    hf_repo: str | None = None
    hf_model_file: str | None = None
    hf_config_file: str | None = None
    sdk_mode: str | None = None
    sdk_model: str | None = None
    command_env_var: str | None = None
    voices_env_var: str | None = None
    voice_designs: dict[str, str] | None = None
    smoke_test_voices: tuple[str, ...] | None = None
    server_url_env_var: str | None = None
    reference_audio_env_var: str | None = None
    reference_text_env_var: str | None = None

    @property
    def local_dir(self) -> Path:
        return TTS_MODELS_ROOT / self.key


DEFAULT_TTS_MODEL_KEY = "valtec_multispeaker"
TIER_A_TTS_MODEL_KEYS = (
    "valtec_multispeaker",
    "mms_tts_vie",
    "piper_vi_vivos_x_low",
    "vieneu_v2_turbo",
    "vieneu_v2_standard",
    "kani_370m_vie",
    "viet_tts_onnx",
)
TIER_B_TTS_MODEL_KEYS = (
    "vixtts",
    "f5_vi_hynt",
    "f5_vi_zalopay",
    "omnivoice",
)

TTS_MODEL_REGISTRY: dict[str, TTSModelConfig] = {
    "valtec_multispeaker": TTSModelConfig(
        key="valtec_multispeaker",
        display_name="Valtec multi-speaker Vietnamese TTS",
        role="baseline",
        runner="valtec",
        default_voice="NF",
        license="CC BY-NC 4.0",
        source_url="https://github.com/tronghieuit/valtec-tts",
        notes="Current SoCa baseline; local CPU runtime; five Vietnamese voices.",
    ),
    "mms_tts_vie": TTSModelConfig(
        key="mms_tts_vie",
        display_name="facebook/mms-tts-vie",
        role="lightweight_baseline",
        runner="mms_transformers",
        default_voice="default",
        license="CC BY-NC 4.0",
        source_url="https://huggingface.co/facebook/mms-tts-vie",
        notes="Single-voice VITS baseline through Transformers; useful lightweight comparison.",
    ),
    "piper_vi_vivos_x_low": TTSModelConfig(
        key="piper_vi_vivos_x_low",
        display_name="Piper vi_VN VIVOS x-low",
        role="edge_fallback",
        runner="piper",
        default_voice="VIVOSSPK13",
        license="unknown in mirror",
        source_url="https://huggingface.co/speaches-ai/piper-vi_VN-vivos-x_low",
        notes="ONNX edge fallback candidate; needs Piper runtime integration.",
        hf_repo="rhasspy/piper-voices",
        hf_model_file="vi/vi_VN/vivos/x_low/vi_VN-vivos-x_low.onnx",
        hf_config_file="vi/vi_VN/vivos/x_low/vi_VN-vivos-x_low.onnx.json",
    ),
    "vieneu_v2_turbo": TTSModelConfig(
        key="vieneu_v2_turbo",
        display_name="VieNeu-TTS v2 Turbo",
        role="primary_challenger",
        runner="vieneu",
        default_voice="XuanVinh",
        license="Apache-2.0",
        source_url="https://huggingface.co/pnnbao-ump/VieNeu-TTS-v2-Turbo",
        notes=(
            "CPU/GGUF Turbo path through the vieneu SDK; primary challenger for SoCa."
        ),
        hf_repo="pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF",
        hf_model_file="vieneu-tts-v2-turbo.gguf",
        sdk_mode="turbo",
    ),
    "vieneu_v2_standard": TTSModelConfig(
        key="vieneu_v2_standard",
        display_name="VieNeu-TTS v2",
        role="quality_runtime_challenger",
        runner="vieneu",
        default_voice="TrucLy",
        license="Apache-2.0",
        source_url="https://huggingface.co/pnnbao-ump/VieNeu-TTS-v2",
        notes="Higher-quality VieNeu GGUF path through the vieneu SDK.",
        hf_repo="pnnbao-ump/VieNeu-TTS-v2",
        hf_model_file="VieNeu-TTS-v2-Q4-K-M.gguf",
        sdk_mode="standard",
    ),
    "kani_370m_vie": TTSModelConfig(
        key="kani_370m_vie",
        display_name="Kani TTS 370M Vie",
        role="expressive_runtime_candidate",
        runner="kani",
        default_voice="default",
        license="Apache-2.0",
        source_url="https://huggingface.co/pnnbao-ump/kani-tts-370m-vie",
        notes="Direct kani-tts-2 runner; voice argument maps to model language tag when exposed.",
        hf_repo="pnnbao-ump/kani-tts-370m-vie",
    ),
    "viet_tts_onnx": TTSModelConfig(
        key="viet_tts_onnx",
        display_name="VietTTS ONNX",
        role="onnx_candidate",
        runner="viettts_server",
        default_voice="cdteam",
        license="Apache-2.0 source; pretrained model/audio CC BY-NC per README",
        source_url="https://github.com/dangvansam/viet-tts",
        notes="OpenAI-compatible local server runner; start VietTTS server before benchmarking.",
        hf_repo="dangvansam/viet-tts",
        server_url_env_var="SHRIKE7_VIET_TTS_BASE_URL",
    ),
    "vixtts": TTSModelConfig(
        key="vixtts",
        display_name="capleaf/viXTTS",
        role="quality_baseline",
        runner="external_command",
        default_voice="default",
        license="Coqui Public Model License",
        source_url="https://huggingface.co/capleaf/viXTTS",
        notes=(
            "Quality baseline. The upstream demo notes Ubuntu/WSL focus and Coqui-library "
            "incompatibility, so SoCa calls a configured external command."
        ),
        hf_repo="capleaf/viXTTS",
        command_env_var="SHRIKE7_TTS_VIXTTS_COMMAND",
        voices_env_var="SHRIKE7_TTS_VIXTTS_VOICES",
    ),
    "f5_vi_hynt": TTSModelConfig(
        key="f5_vi_hynt",
        display_name="F5-TTS Vietnamese ViVoice",
        role="quality_baseline",
        runner="f5",
        default_voice="reference",
        license="CC BY-NC-SA 4.0",
        source_url="https://huggingface.co/hynt/F5-TTS-Vietnamese-ViVoice",
        notes="F5-TTS quality baseline; requires a local reference WAV and transcript.",
        hf_repo="hynt/F5-TTS-Vietnamese-ViVoice",
        hf_model_file="model_last.pt",
        hf_config_file="config.json",
        sdk_model="F5TTS_Base",
        reference_audio_env_var="SHRIKE7_TTS_F5_HYNT_REF_AUDIO",
        reference_text_env_var="SHRIKE7_TTS_F5_HYNT_REF_TEXT",
    ),
    "f5_vi_zalopay": TTSModelConfig(
        key="f5_vi_zalopay",
        display_name="ZaloPay Vietnamese F5-TTS",
        role="secondary_quality_baseline",
        runner="f5",
        default_voice="reference",
        license="CC-BY-4.0 metadata; verify model terms before redistribution",
        source_url="https://huggingface.co/zalopay/vietnamese-tts",
        notes="Secondary F5-TTS quality baseline; requires a local reference WAV and transcript.",
        hf_repo="zalopay/vietnamese-tts",
        hf_model_file="model_1290000.pt",
        hf_config_file="vocab.txt",
        sdk_model="F5TTS_Base",
        reference_audio_env_var="SHRIKE7_TTS_F5_ZALOPAY_REF_AUDIO",
        reference_text_env_var="SHRIKE7_TTS_F5_ZALOPAY_REF_TEXT",
    ),
    "omnivoice": TTSModelConfig(
        key="omnivoice",
        display_name="k2-fsa/OmniVoice",
        role="quality_baseline",
        runner="omnivoice",
        default_voice="auto",
        license="Apache-2.0",
        source_url="https://huggingface.co/k2-fsa/OmniVoice",
        notes=(
            "Massively multilingual zero-shot TTS baseline; heavy but direct Python API exists. "
            "Registry default uses auto voice-design mode; saved voices are selected explicitly."
        ),
        hf_repo="k2-fsa/OmniVoice",
        voices_env_var="SHRIKE7_TTS_OMNIVOICE_VOICES",
        voice_designs={
            "female_young": "female, young adult, moderate pitch",
            "female_high": "female, young adult, high pitch",
            "male_low": "male, middle-aged, low pitch",
            "male_deep": "male, middle-aged, very low pitch",
        },
        smoke_test_voices=("female_young", "female_high", "male_low", "male_deep"),
    ),
}


def get_tts_model_config(model_key: str) -> TTSModelConfig:
    try:
        return TTS_MODEL_REGISTRY[model_key]
    except KeyError as exc:
        valid = ", ".join(sorted(TTS_MODEL_REGISTRY))
        raise ValueError(f"Unknown TTS model key: {model_key}. Valid keys: {valid}") from exc
