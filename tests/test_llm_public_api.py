from __future__ import annotations

from soca.llm import LLMEngine, LLMResult, LocalLlamaCppLLM
from soca.llm.registry import DEFAULT_LLM_MODEL_KEY, get_model_config


def test_llm_package_exports_generic_runtime_types_only() -> None:
    assert LLMEngine.__name__ == "LLMEngine"
    assert LLMResult.__name__ == "LLMResult"
    assert LocalLlamaCppLLM.__name__ == "LocalLlamaCppLLM"


def test_default_llamacpp_model_is_registry_default() -> None:
    config = get_model_config(DEFAULT_LLM_MODEL_KEY)

    assert DEFAULT_LLM_MODEL_KEY == "phogpt_4b_q4_k_m"
    assert config.local_path.name == "PhoGPT-4B-Chat-Q4_K_M.gguf"
