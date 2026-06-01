
from dataclasses import fields

import pytest

from soca.llm.registry import (
    BAKEOFF_MODEL_KEYS,
    DEFAULT_LLM_MODEL_KEY,
    FULL_MODEL_KEYS,
    LLM_MODEL_REGISTRY,
    MODELS_ROOT,
    get_model_config,
    get_profile_configs,
    get_profile_model_keys,
)


def test_default_model_key_exists() -> None:
    assert DEFAULT_LLM_MODEL_KEY in LLM_MODEL_REGISTRY


def test_registry_keys_match_config_model_keys() -> None:
    for key, config in LLM_MODEL_REGISTRY.items():
        assert config.model_key == key


def test_bakeoff_and_full_keys_exist() -> None:
    for key in BAKEOFF_MODEL_KEYS + FULL_MODEL_KEYS:
        assert key in LLM_MODEL_REGISTRY


def test_model_paths_are_under_models_root() -> None:
    models_root = MODELS_ROOT.resolve()
    for config in LLM_MODEL_REGISTRY.values():
        local_path = config.local_path.resolve()
        assert local_path.is_relative_to(models_root)


def test_get_model_config_returns_config() -> None:
    config = get_model_config(DEFAULT_LLM_MODEL_KEY)
    assert config.hf_repo == "vinai/PhoGPT-4B-Chat-gguf"


def test_get_model_config_rejects_unknown_key() -> None:
    with pytest.raises(KeyError, match="Unknown LLM model key"):
        get_model_config("not_a_real_model")


def test_profile_minimal_contains_default_only() -> None:
    assert get_profile_model_keys("minimal") == [DEFAULT_LLM_MODEL_KEY]


def test_profile_bakeoff_returns_configs() -> None:
    configs = get_profile_configs("bakeoff")
    assert [config.model_key for config in configs] == BAKEOFF_MODEL_KEYS


def test_unknown_profile_has_helpful_error() -> None:
    with pytest.raises(KeyError, match="Unknown LLM profile"):
        get_profile_model_keys("not_a_profile")


def test_download_command_points_to_generic_downloader() -> None:
    config = get_model_config(DEFAULT_LLM_MODEL_KEY)
    assert config.download_command == (
        "uv run python scripts/download_llm.py --model phogpt_4b_q4_k_m"
    )


def test_registry_has_prompt_runtime_metadata_fields() -> None:
    field_names = {field.name for field in fields(type(get_model_config(DEFAULT_LLM_MODEL_KEY)))}

    assert "chat_format" in field_names
    assert "stop_sequences" in field_names
    assert "append_no_think" in field_names
    assert "strip_reasoning" in field_names


def test_phogpt_has_completion_stop_sequence() -> None:
    config = get_model_config("phogpt_4b_q4_k_m")

    assert config.prompt_style == "phogpt_completion"
    assert config.chat_format is None
    assert config.stop_sequences == ("### Câu hỏi:",)
    assert config.append_no_think is False
    assert config.strip_reasoning is False


def test_qwen3_tiny_disables_thinking_for_voice_assistant_use() -> None:
    config = get_model_config("qwen3_0_6b_q8_0")

    assert config.prompt_style == "qwen_chat_no_think"
    assert config.chat_format is None
    assert config.stop_sequences == ()
    assert config.append_no_think is True
    assert config.strip_reasoning is True


def test_all_qwen_no_think_models_have_runtime_safeguards() -> None:
    for config in LLM_MODEL_REGISTRY.values():
        if config.prompt_style == "qwen_chat_no_think":
            assert config.append_no_think is True
            assert config.strip_reasoning is True


def test_vinallama_27b_uses_chatml_fallback() -> None:
    config = get_model_config("vinallama_2_7b_q5_0")

    assert config.prompt_style == "llama_chat"
    assert config.chat_format == "chatml"
    assert config.stop_sequences == ()


def test_bkai_probe_uses_llama2_chat_fallback() -> None:
    config = get_model_config("bkai_llama2_120gb_q4_k_m")

    assert config.prompt_style == "llama_completion_or_chat_probe"
    assert config.chat_format == "llama-2"
    assert config.stop_sequences == ()
