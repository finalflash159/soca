from __future__ import annotations

import pytest

from soca.core.context_budget import (
    ModelCapability,
    PromptAssembler,
    PromptBudgetError,
    PromptComponent,
)


def test_assembler_clamps_output_and_drops_low_priority_optional_context() -> None:
    assembler = PromptAssembler(
        ModelCapability("small", context_window=256),
        safety_margin_tokens=16,
        minimum_input_tokens=32,
    )
    prompt, manifest = assembler.assemble(
        [
            PromptComponent("system", "system", priority=0, required=True),
            PromptComponent("current", "question", priority=0, required=True),
            PromptComponent("knowledge", "k" * 200, priority=1),
            PromptComponent("archive", "x" * 2_000, priority=50),
        ],
        requested_output_tokens=500,
    )

    assert "system" in prompt
    assert "question" in prompt
    assert manifest.effective_output_tokens == 208
    assert manifest.input_budget_tokens == 32
    assert manifest.safety_margin_tokens == 16
    assert manifest.dropped_components == ("knowledge", "archive")
    assert manifest.prompt_tokens <= manifest.input_budget_tokens


def test_required_context_overflow_fails_before_model_call() -> None:
    assembler = PromptAssembler(
        ModelCapability("tiny", context_window=128),
        safety_margin_tokens=16,
        minimum_input_tokens=16,
    )

    with pytest.raises(PromptBudgetError, match="required component"):
        assembler.assemble(
            [PromptComponent("system", "s" * 500, priority=0, required=True)],
            requested_output_tokens=100,
        )


def test_manifest_hash_is_deterministic_and_unknown_context_is_reported() -> None:
    assembler = PromptAssembler(ModelCapability("remote/model", context_window=None))
    components = [PromptComponent("current", "xin chào", priority=0, required=True)]

    _, first = assembler.assemble(components, requested_output_tokens=4_096)
    _, second = assembler.assemble(components, requested_output_tokens=4_096)

    assert first.prompt_hash == second.prompt_hash
    assert first.context_window is None
    assert first.input_budget_tokens is None


@pytest.mark.parametrize("context_window", [2_048, 4_096, 16_384, 32_768])
def test_context_window_matrix_keeps_required_input_and_clamps_output(
    context_window: int,
) -> None:
    assembler = PromptAssembler(ModelCapability("matrix", context_window=context_window))
    _, manifest = assembler.assemble(
        [
            PromptComponent("system", "system instructions", priority=0, required=True),
            PromptComponent("goal", "goal and success criteria", priority=0, required=True),
            PromptComponent("current_input", "câu hỏi hiện tại", priority=0, required=True),
            PromptComponent("archive", "archive context " * 2_000, priority=50),
        ],
        requested_output_tokens=4_096,
    )

    required_ids = {"system", "goal", "current_input"}
    included_ids = {
        item.component_id for item in manifest.components if item.included
    }
    assert required_ids <= included_ids
    assert manifest.effective_output_tokens <= 4_096
    assert manifest.prompt_tokens <= manifest.input_budget_tokens
    assert manifest.prompt_tokens + manifest.effective_output_tokens <= context_window - 32
