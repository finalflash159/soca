from __future__ import annotations

from scripts.download_llm import format_bytes, select_configs
from soca.llm.registry import DEFAULT_LLM_MODEL_KEY


def test_select_configs_defaults_to_minimal_profile() -> None:
    configs = select_configs([], None)

    assert [config.model_key for config in configs] == [DEFAULT_LLM_MODEL_KEY]


def test_select_configs_combines_profile_and_explicit_models_without_duplicates() -> None:
    configs = select_configs(
        ["phogpt_4b_q4_k_m", "qwen3_4b_q4_k_m"],
        "bakeoff",
    )

    assert [config.model_key for config in configs] == [
        "phogpt_4b_q4_k_m",
        "arcee_vylinh_3b_q4_k_m",
        "qwen3_0_6b_q8_0",
        "vinallama_2_7b_q5_0",
        "qwen3_4b_q4_k_m",
    ]


def test_format_bytes_uses_human_units() -> None:
    assert format_bytes(512) == "512 B"
    assert format_bytes(2048) == "2.00 KB"
    assert format_bytes(5 * 1024 * 1024) == "5.00 MB"
