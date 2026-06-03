import pytest

from soca.llm.message_format import (
    build_chat_messages,
    build_completion_prompt,
    uses_completion_prompt,
)
from soca.llm.output_cleaning import StreamingOutputCleaner, clean_model_output
from soca.llm.registry import get_model_config
from soca.prompts import SOCA_RUNTIME_SYSTEM_PROMPT, build_runtime_prompt


def test_phogpt_completion_prompt_preserves_template() -> None:
    config = get_model_config("phogpt_4b_q4_k_m")

    prompt = build_completion_prompt("Xin chào", config, inject_persona=False)

    assert prompt == "### Câu hỏi: Xin chào\n### Trả lời:"


def test_phogpt_completion_prompt_injects_persona() -> None:
    config = get_model_config("phogpt_4b_q4_k_m")

    prompt = build_completion_prompt("Bạn là ai?", config, inject_persona=True)

    assert "Bạn là Sơn Ca" in prompt
    assert "Câu hỏi của tôi: Bạn là ai?" in prompt


def test_completion_prompt_rejects_chat_model() -> None:
    config = get_model_config("arcee_vylinh_3b_q4_k_m")

    with pytest.raises(ValueError, match="does not use completion"):
        build_completion_prompt("Xin chào", config)


def test_chat_messages_include_system_and_user() -> None:
    config = get_model_config("arcee_vylinh_3b_q4_k_m")

    messages = build_chat_messages("Xin chào", config, inject_persona=True)

    assert messages[0]["role"] == "system"
    assert "Bạn là Sơn Ca" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "Xin chào"}


def test_chat_messages_do_not_hand_render_chat_tokens() -> None:
    config = get_model_config("arcee_vylinh_3b_q4_k_m")

    messages = build_chat_messages("Xin chào", config, inject_persona=True)
    combined = "\n".join(message["content"] for message in messages)

    assert "<|im_start|>" not in combined
    assert "[INST]" not in combined
    assert "<start_of_turn>" not in combined


def test_qwen3_appends_no_think_marker() -> None:
    config = get_model_config("qwen3_0_6b_q8_0")

    messages = build_chat_messages("Giải thích GPU là gì", config, inject_persona=True)

    assert messages[-1]["content"].endswith("/no_think")


def test_qwen3_does_not_duplicate_no_think_marker() -> None:
    config = get_model_config("qwen3_0_6b_q8_0")

    messages = build_chat_messages("Giải thích GPU là gì\n/no_think", config)

    assert messages[-1]["content"].count("/no_think") == 1


def test_chat_messages_promote_runtime_prompt_to_system_role() -> None:
    config = get_model_config("arcee_vylinh_3b_q4_k_m")
    prompt = build_runtime_prompt(
        user_text="Xin chào",
        memory_prompt_text="Recent conversation:\nUser: tôi thích trả lời ngắn",
    )

    messages = build_chat_messages(prompt, config, inject_persona=False)

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SOCA_RUNTIME_SYSTEM_PROMPT.strip()
    assert messages[1]["role"] == "user"
    assert "Recent conversation:" in messages[1]["content"]
    assert "Câu hỏi hiện tại:" in messages[1]["content"]
    assert "Bạn là Sơn Ca" not in messages[1]["content"]


def test_chat_messages_promote_runtime_prompt_before_qwen_no_think_marker() -> None:
    config = get_model_config("qwen3_0_6b_q8_0")
    prompt = build_runtime_prompt(user_text="Xin chào")

    messages = build_chat_messages(prompt, config, inject_persona=False)

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SOCA_RUNTIME_SYSTEM_PROMPT.strip()
    assert messages[1]["content"].endswith("/no_think")
    assert "Bạn là Sơn Ca" not in messages[1]["content"]


def test_chat_messages_reject_phogpt() -> None:
    config = get_model_config("phogpt_4b_q4_k_m")

    with pytest.raises(ValueError, match="completion prompts"):
        build_chat_messages("Xin chào", config)


def test_clean_model_output_strips_reasoning_when_enabled() -> None:
    config = get_model_config("qwen3_0_6b_q8_0")
    text = "<think>tôi cần suy nghĩ</think>\nXin chào bạn."

    assert clean_model_output(text, config) == "Xin chào bạn."


def test_clean_model_output_keeps_angle_text_when_reasoning_disabled() -> None:
    config = get_model_config("arcee_vylinh_3b_q4_k_m")
    text = "<tag>Xin chào</tag>"

    assert clean_model_output(text, config) == "<tag>Xin chào</tag>"


def test_streaming_output_cleaner_strips_reasoning_across_chunks() -> None:
    config = get_model_config("qwen3_0_6b_q8_0")
    cleaner = StreamingOutputCleaner(config)

    output: list[str] = []
    for chunk in ["<thi", "nk>ẩn", " nội bộ</thi", "nk>\n\nXin", " chào!"]:
        output.extend(cleaner.feed(chunk))
    output.extend(cleaner.flush())

    assert "".join(output) == "Xin chào!"


def test_streaming_output_cleaner_keeps_angle_text_when_reasoning_disabled() -> None:
    config = get_model_config("arcee_vylinh_3b_q4_k_m")
    cleaner = StreamingOutputCleaner(config)

    output = cleaner.feed("<tag>Xin chào</tag>") + cleaner.flush()

    assert output == ["<tag>Xin chào</tag>"]


def test_uses_completion_prompt() -> None:
    assert uses_completion_prompt(get_model_config("phogpt_4b_q4_k_m")) is True
    assert uses_completion_prompt(get_model_config("arcee_vylinh_3b_q4_k_m")) is False
