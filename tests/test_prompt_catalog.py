from __future__ import annotations

from soca.prompts import (
    SOCA_RUNTIME_SYSTEM_PROMPT,
    build_runtime_prompt,
    split_embedded_system_prompt,
)


def test_runtime_prompt_uses_soca_and_bounded_context_sections() -> None:
    prompt = build_runtime_prompt(
        user_text="Theo wiki, bữa sáng nên ăn gì?",
        memory_prompt_text="Long-term memory:\n- thích giải thích kỹ",
        knowledge_prompt_text="[K1] wiki/dinh-duong/bua-sang.md\nCó đạm và rau.",
    )

    assert "Bạn là SoCa" in prompt
    assert "Memory:" in prompt
    assert "Knowledge:" in prompt
    assert "[K1]" in prompt
    assert "Câu hỏi hiện tại:" in prompt
    assert "Trả lời:" in prompt


def test_split_embedded_system_prompt_extracts_runtime_system_block() -> None:
    prompt = build_runtime_prompt(
        user_text="Xin chào",
        memory_prompt_text="Long-term memory:\n- thích ngắn gọn",
    )

    system_prompt, user_content = split_embedded_system_prompt(prompt)

    assert system_prompt == SOCA_RUNTIME_SYSTEM_PROMPT.strip()
    assert "Memory:" in user_content
    assert "Câu hỏi hiện tại:" in user_content
    assert "Bạn là SoCa" not in user_content


def test_split_embedded_system_prompt_leaves_plain_user_text_alone() -> None:
    system_prompt, user_content = split_embedded_system_prompt("Xin chào")

    assert system_prompt is None
    assert user_content == "Xin chào"
