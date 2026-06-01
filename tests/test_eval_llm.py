from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.eval_llm import (
    count_words,
    format_optional_rate,
    has_cjk_leak,
    has_english_leak,
    has_uncertainty_or_refusal,
    has_vietnamese_signal,
    is_command_ack,
    is_command_refusal,
    is_privacy_safe,
    is_privacy_sensitive,
    is_realtime_safe,
    is_realtime_sensitive,
    load_prompts,
    parse_model_list,
    rate,
    select_model_keys,
    summarize,
)
from soca.llm.registry import DEFAULT_LLM_MODEL_KEY


def test_select_model_keys_defaults_to_registry_default() -> None:
    assert select_model_keys([], None) == [DEFAULT_LLM_MODEL_KEY]


def test_select_model_keys_dedupes_profile_and_explicit_models() -> None:
    assert select_model_keys(["phogpt_4b_q4_k_m", "qwen3_4b_q4_k_m"], "bakeoff") == [
        "phogpt_4b_q4_k_m",
        "arcee_vylinh_3b_q4_k_m",
        "qwen3_0_6b_q8_0",
        "vinallama_2_7b_q5_0",
        "qwen3_4b_q4_k_m",
    ]


def test_parse_model_list_accepts_comma_separated_values() -> None:
    assert parse_model_list(["phogpt_4b_q4_k_m,qwen3_0_6b_q8_0", "vinallama_2_7b_q5_0"]) == [
        "phogpt_4b_q4_k_m",
        "qwen3_0_6b_q8_0",
        "vinallama_2_7b_q5_0",
    ]


def test_parse_model_list_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unknown LLM model key"):
        parse_model_list(["not_real"])


def test_load_prompts_reads_jsonl_and_respects_limit(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "category": "x", "prompt": "Xin chào"}, ensure_ascii=False),
                json.dumps(
                    {
                        "id": "b",
                        "category": "y",
                        "prompt": "Tạm biệt",
                        "expected_behavior": ["vietnamese"],
                        "max_words": 12,
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    prompts = load_prompts(prompt_path, limit=1)

    assert len(prompts) == 1
    assert prompts[0].prompt_id == "a"
    assert prompts[0].text == "Xin chào"
    assert prompts[0].expected_behavior == ()
    assert prompts[0].max_words == 50


def test_load_prompts_reads_behavior_metadata(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        json.dumps(
            {
                "id": "a",
                "category": "x",
                "prompt": "Xin chào",
                "expected_behavior": ["vietnamese", "concise"],
                "max_words": 10,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    prompt = load_prompts(prompt_path)[0]

    assert prompt.expected_behavior == ("vietnamese", "concise")
    assert prompt.max_words == 10


def test_voice_assistant_output_heuristics() -> None:
    assert count_words("Xin chào bạn") == 3
    assert has_vietnamese_signal("Xin chào bạn") is True
    assert has_vietnamese_signal("Hello world") is False
    assert has_cjk_leak("Tôi ăn cơm昨日") is True
    assert has_cjk_leak("Tôi ăn cơm hôm qua") is False
    assert has_uncertainty_or_refusal("Xin lỗi, tôi không thể biết thời gian.") is True


def test_behavior_metrics_for_command_prompts() -> None:
    prompt = load_prompts(Path("eval/prompts/llm_bakeoff_vi.jsonl"))[0]

    assert prompt.category == "assistant_command"
    assert is_command_refusal(prompt, "Xin lỗi, tôi không thể đặt lời nhắc.") is True
    assert is_command_ack(prompt, "Được, tôi sẽ nhắc bạn lúc 8 giờ tối.") is True
    assert is_command_ack(prompt, "Xin lỗi, tôi không thể đặt lời nhắc.") is False


def test_behavior_metrics_for_realtime_and_privacy_prompts() -> None:
    realtime_prompt = next(
        prompt for prompt in load_prompts(Path("eval/prompts/llm_bakeoff_vi.jsonl"))
        if prompt.prompt_id == "local_002"
    )
    privacy_prompt = next(
        prompt for prompt in load_prompts(Path("eval/prompts/llm_bakeoff_vi.jsonl"))
        if prompt.prompt_id == "local_008"
    )

    assert is_realtime_sensitive(realtime_prompt) is True
    assert is_realtime_safe(
        realtime_prompt,
        "Tôi không có dữ liệu thời tiết thực tế để trả lời chính xác.",
    ) is True
    assert is_realtime_safe(realtime_prompt, "Hôm nay trời nắng đẹp.") is False
    assert is_privacy_sensitive(privacy_prompt) is True
    assert is_privacy_safe(privacy_prompt, "Không nên chia sẻ mã OTP cho bất kỳ ai.") is True


def test_english_leak_ignores_coding_prompts() -> None:
    chat_prompt = next(
        prompt for prompt in load_prompts(Path("eval/prompts/llm_bakeoff_vi.jsonl"))
        if prompt.prompt_id == "chat_009"
    )
    coding_prompt = next(
        prompt for prompt in load_prompts(Path("eval/prompts/llm_bakeoff_vi.jsonl"))
        if prompt.prompt_id == "code_002"
    )

    assert has_english_leak("I don't know.", chat_prompt) is True
    assert has_english_leak("def add(a, b): return a + b", coding_prompt) is False


def test_rate_helpers() -> None:
    assert rate([True, False, True]) == pytest.approx(2 / 3)
    assert rate([]) is None
    assert format_optional_rate(None) == "n/a"
    assert format_optional_rate(0.25) == "25.0%"


def test_summarize_returns_basic_distribution() -> None:
    stats = summarize([1, 2, 3, 4])

    assert stats["mean"] == 2.5
    assert stats["median"] == 2.5
    assert stats["min"] == 1
    assert stats["max"] == 4
