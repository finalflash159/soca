from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT / "models"


@dataclass(frozen=True)
class LLMModelConfig:
    model_key: str
    hf_repo: str
    filename: str
    local_dir_name: str
    prompt_style: str
    role: str
    context_window: int
    license_note: str
    source_url: str
    # Fallback for llama-cpp-python when GGUF metadata does not provide a usable chat template.
    chat_format: str | None = None
    stop_sequences: tuple[str, ...] = ()
    # Voice assistant safeguard for reasoning-capable models such as Qwen3.
    append_no_think: bool = False
    strip_reasoning: bool = False

    @property
    def local_dir(self) -> Path:
        return MODELS_ROOT / self.local_dir_name

    @property
    def local_path(self) -> Path:
        return self.local_dir / self.filename

    @property
    def download_command(self) -> str:
        return f"uv run python scripts/download_llm.py --model {self.model_key}"


DEFAULT_LLM_MODEL_KEY = "phogpt_4b_q4_k_m"

LLM_MODEL_REGISTRY: dict[str, LLMModelConfig] = {
    "phogpt_4b_q4_k_m": LLMModelConfig(
        model_key="phogpt_4b_q4_k_m",
        hf_repo="vinai/PhoGPT-4B-Chat-gguf",
        filename="PhoGPT-4B-Chat-Q4_K_M.gguf",
        local_dir_name="phogpt-4b-chat-gguf",
        prompt_style="phogpt_completion",
        role="baseline",
        context_window=2048,
        license_note="BSD-3-Clause model card; keep as current Soca baseline.",
        source_url="https://huggingface.co/vinai/PhoGPT-4B-Chat-gguf",
        stop_sequences=("### Câu hỏi:",),
    ),
    "arcee_vylinh_3b_q4_k_m": LLMModelConfig(
        model_key="arcee_vylinh_3b_q4_k_m",
        hf_repo="QuantFactory/Arcee-VyLinh-GGUF",
        filename="Arcee-VyLinh.Q4_K_M.gguf",
        local_dir_name="arcee-vylinh-gguf",
        prompt_style="qwen_chat",
        role="primary_candidate",
        context_window=32768,
        license_note="Community GGUF of Arcee-VyLinh; verify upstream license before public release claims.",
        source_url="https://huggingface.co/QuantFactory/Arcee-VyLinh-GGUF",
    ),
    "qwen3_0_6b_q8_0": LLMModelConfig(
        model_key="qwen3_0_6b_q8_0",
        hf_repo="Qwen/Qwen3-0.6B-GGUF",
        filename="Qwen3-0.6B-Q8_0.gguf",
        local_dir_name="qwen3-0.6b-gguf",
        prompt_style="qwen_chat_no_think",
        role="low_ram_fallback",
        context_window=32768,
        license_note="Apache-2.0; tiny fallback for low-memory tests.",
        source_url="https://huggingface.co/Qwen/Qwen3-0.6B-GGUF",
        append_no_think=True,
        strip_reasoning=True,
    ),
    "vinallama_2_7b_q5_0": LLMModelConfig(
        model_key="vinallama_2_7b_q5_0",
        hf_repo="vilm/vinallama-2.7b-chat-GGUF",
        filename="vinallama-2.7b-chat_q5_0.gguf",
        local_dir_name="vinallama-2.7b-chat-gguf",
        prompt_style="llama_chat",
        role="vietnamese_trained_small_candidate",
        context_window=4096,
        license_note="Llama2-family license; useful Vietnamese-trained small candidate.",
        source_url="https://huggingface.co/vilm/vinallama-2.7b-chat-GGUF",
        chat_format="chatml",
    ),
    "qwen3_4b_q4_k_m": LLMModelConfig(
        model_key="qwen3_4b_q4_k_m",
        hf_repo="unsloth/Qwen3-4B-Instruct-2507-GGUF",
        filename="Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        local_dir_name="qwen3-4b-instruct-2507-gguf",
        prompt_style="qwen_chat_no_think",
        role="modern_multilingual_quality_candidate",
        context_window=32768,
        license_note="Apache-2.0 upstream; modern multilingual quality candidate.",
        source_url="https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF",
        append_no_think=True,
        strip_reasoning=True,
    ),
    "sailor2_1b_q4_0": LLMModelConfig(
        model_key="sailor2_1b_q4_0",
        hf_repo="bartowski/Sailor2-1B-Chat-GGUF",
        filename="Sailor2-1B-Chat-Q4_0.gguf",
        local_dir_name="sailor2-1b-chat-gguf",
        prompt_style="qwen_chat",
        role="sea_tiny_fallback",
        context_window=4096,
        license_note="Apache-2.0 upstream; SEA language fallback candidate.",
        source_url="https://huggingface.co/bartowski/Sailor2-1B-Chat-GGUF",
    ),
    "vistral_7b_q4_0": LLMModelConfig(
        model_key="vistral_7b_q4_0",
        hf_repo="KhanhVan/Vistral-7B-Chat-gguf",
        filename="ggml-vistral-7B-chat-q4_0.gguf",
        local_dir_name="vistral-7b-chat-gguf",
        prompt_style="mistral_chat",
        role="vietnamese_quality_ceiling",
        context_window=4096,
        license_note="AFL-3.0 card signal; verify before public release claims.",
        source_url="https://huggingface.co/KhanhVan/Vistral-7B-Chat-gguf",
    ),
    "vinallama_7b_q5_0": LLMModelConfig(
        model_key="vinallama_7b_q5_0",
        hf_repo="vilm/vinallama-7b-chat-GGUF",
        filename="vinallama-7b-chat_q5_0.gguf",
        local_dir_name="vinallama-7b-chat-gguf",
        prompt_style="llama_chat",
        role="vietnamese_quality_ceiling",
        context_window=4096,
        license_note="Llama2-family license; heavier Vietnamese-trained quality candidate.",
        source_url="https://huggingface.co/vilm/vinallama-7b-chat-GGUF",
    ),
    "bkai_llama2_120gb_q4_k_m": LLMModelConfig(
        model_key="bkai_llama2_120gb_q4_k_m",
        hf_repo="nold/vietnamese-llama2-7b-120GB-GGUF",
        filename="vietnamese-llama2-7b-120GB_Q4_K_M.gguf",
        local_dir_name="vietnamese-llama2-7b-120gb-gguf",
        prompt_style="llama_completion_or_chat_probe",
        role="vietnamese_fluency_probe",
        context_window=4096,
        license_note="Research probe; likely needs SFT for assistant behavior.",
        source_url="https://huggingface.co/nold/vietnamese-llama2-7b-120GB-GGUF",
        chat_format="llama-2",
    ),
    "dataops_qwen25_7b_vi_q4_k_m": LLMModelConfig(
        model_key="dataops_qwen25_7b_vi_q4_k_m",
        hf_repo="DataOpsFusion/Qwen2.5-7B-Instruct-vietnamese-r32-GGUF",
        filename="Qwen2.5-7B-Instruct-vietnamese-r32-Q4_K_M.gguf",
        local_dir_name="qwen2.5-7b-instruct-vietnamese-r32-gguf",
        prompt_style="qwen_chat",
        role="vietnamese_finetune_probe",
        context_window=32768,
        license_note="Vietnamese finetune probe; low HF signal, benchmark before trusting.",
        source_url="https://huggingface.co/DataOpsFusion/Qwen2.5-7B-Instruct-vietnamese-r32-GGUF",
    ),
    "dataops_gemma3_4b_vi_q4_k_m": LLMModelConfig(
        model_key="dataops_gemma3_4b_vi_q4_k_m",
        hf_repo="DataOpsFusion/gemma-3-4b-it-vietnamese-r16-GGUF",
        filename="gemma-3-4b-it-vietnamese-r16-Q4_K_M.gguf",
        local_dir_name="gemma-3-4b-it-vietnamese-r16-gguf",
        prompt_style="gemma_chat",
        role="vietnamese_finetune_probe",
        context_window=8192,
        license_note="Vietnamese Gemma probe; test runtime compatibility carefully.",
        source_url="https://huggingface.co/DataOpsFusion/gemma-3-4b-it-vietnamese-r16-GGUF",
    ),
}

BAKEOFF_MODEL_KEYS = [
    "phogpt_4b_q4_k_m",
    "arcee_vylinh_3b_q4_k_m",
    "qwen3_0_6b_q8_0",
    "vinallama_2_7b_q5_0",
]

FULL_MODEL_KEYS = BAKEOFF_MODEL_KEYS + [
    "qwen3_4b_q4_k_m",
    "sailor2_1b_q4_0",
    "vistral_7b_q4_0",
    "vinallama_7b_q5_0",
    "bkai_llama2_120gb_q4_k_m",
    "dataops_qwen25_7b_vi_q4_k_m",
    "dataops_gemma3_4b_vi_q4_k_m",
]

PROFILE_MODEL_KEYS = {
    "minimal": [DEFAULT_LLM_MODEL_KEY],
    "bakeoff": BAKEOFF_MODEL_KEYS,
    "full": FULL_MODEL_KEYS,
}


def get_model_config(model_key: str) -> LLMModelConfig:
    try:
        return LLM_MODEL_REGISTRY[model_key]
    except KeyError as exc:
        valid = ", ".join(sorted(LLM_MODEL_REGISTRY))
        raise KeyError(f"Unknown LLM model key: {model_key}. Valid keys: {valid}") from exc


def get_profile_model_keys(profile: str) -> list[str]:
    try:
        return list(PROFILE_MODEL_KEYS[profile])
    except KeyError as exc:
        valid = ", ".join(sorted(PROFILE_MODEL_KEYS))
        raise KeyError(f"Unknown LLM profile: {profile}. Valid profiles: {valid}") from exc


def get_profile_configs(profile: str) -> list[LLMModelConfig]:
    return [get_model_config(model_key) for model_key in get_profile_model_keys(profile)]
