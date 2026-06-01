from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT / "models"


@dataclass(frozen=True)
class ASRModelConfig:
    model_key: str
    hf_repo: str
    local_dir_name: str
    params_m: int
    role: str

    @property
    def local_dir(self) -> Path:
        return MODELS_ROOT / self.local_dir_name

    @property
    def onnx_dir(self) -> Path:
        return self.local_dir / "onnx"

    @property
    def encoder_path(self) -> Path:
        return self.onnx_dir / "encoder_model.onnx"

    @property
    def decoder_path(self) -> Path:
        return self.onnx_dir / "decoder_model.onnx"

    @property
    def download_command(self) -> str:
        return f"uv run python scripts/download_phowhisper.py --model {self.model_key}"


DEFAULT_ASR_MODEL_KEY = "phowhisper_tiny"

ASR_MODEL_REGISTRY: dict[str, ASRModelConfig] = {
    "phowhisper_tiny": ASRModelConfig(
        model_key="phowhisper_tiny",
        hf_repo="huuquyet/PhoWhisper-tiny",
        local_dir_name="phowhisper-tiny-onnx",
        params_m=39,
        role="edge_baseline",
    ),
    "phowhisper_base": ASRModelConfig(
        model_key="phowhisper_base",
        hf_repo="huuquyet/PhoWhisper-base",
        local_dir_name="phowhisper-base-onnx",
        params_m=74,
        role="small_quality_candidate",
    ),
    "phowhisper_small": ASRModelConfig(
        model_key="phowhisper_small",
        hf_repo="huuquyet/PhoWhisper-small",
        local_dir_name="phowhisper-small-onnx",
        params_m=244,
        role="mid_quality_candidate",
    ),
    "phowhisper_medium": ASRModelConfig(
        model_key="phowhisper_medium",
        hf_repo="huuquyet/PhoWhisper-medium",
        local_dir_name="phowhisper-medium-onnx",
        params_m=769,
        role="quality_ceiling_probe",
    ),
}

ASR_BAKEOFF_MODEL_KEYS = [
    "phowhisper_tiny",
    "phowhisper_base",
    "phowhisper_small",
]

ASR_FULL_MODEL_KEYS = [
    *ASR_BAKEOFF_MODEL_KEYS,
    "phowhisper_medium",
]

ASR_PROFILE_MODEL_KEYS = {
    "minimal": [DEFAULT_ASR_MODEL_KEY],
    "bakeoff": ASR_BAKEOFF_MODEL_KEYS,
    "full": ASR_FULL_MODEL_KEYS,
}

PHOWHISPER_ALLOW_PATTERNS = [
    "onnx/encoder_model.onnx",
    "onnx/decoder_model.onnx",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "normalizer.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "quantize_config.json",
]


def get_asr_model_config(model_key: str) -> ASRModelConfig:
    try:
        return ASR_MODEL_REGISTRY[model_key]
    except KeyError as exc:
        valid = ", ".join(sorted(ASR_MODEL_REGISTRY))
        raise KeyError(f"Unknown ASR model key: {model_key}. Valid keys: {valid}") from exc


def get_asr_profile_model_keys(profile: str) -> list[str]:
    try:
        return list(ASR_PROFILE_MODEL_KEYS[profile])
    except KeyError as exc:
        valid = ", ".join(sorted(ASR_PROFILE_MODEL_KEYS))
        raise KeyError(f"Unknown ASR profile: {profile}. Valid profiles: {valid}") from exc
