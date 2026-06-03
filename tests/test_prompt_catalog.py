from __future__ import annotations

from soca.prompts import build_memory_aware_prompt, build_runtime_prompt


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


def test_memory_prompt_uses_same_soca_catalog_name() -> None:
    prompt = build_memory_aware_prompt(
        user_text="Bạn nhớ gì về tôi?",
        memory_prompt_text="Long-term memory:\n- thích tiếng Việt",
    )

    assert "Bạn là SoCa" in prompt
    assert "Sơn Ca" not in prompt
    assert "thích tiếng Việt" in prompt
