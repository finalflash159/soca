from __future__ import annotations

import pytest

from soca.core.context_budget import (
    ModelCapability,
    PromptAssembler,
    PromptBudgetError,
    PromptComponent,
    capability_from_engine,
    capability_from_values,
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


def test_manifest_records_capability_provenance() -> None:
    capability = capability_from_values(
        model_id="remote/model",
        context_window=8_192,
        max_output_tokens=2_048,
        tokenizer="provider_estimate",
        source="remote_catalog",
    )
    _, manifest = PromptAssembler(capability).assemble(
        [PromptComponent("current", "xin chào", priority=0, required=True)],
        requested_output_tokens=4_096,
    )

    payload = manifest.to_dict()
    assert payload["capability_source"] == "remote_catalog"
    assert payload["tokenizer"] == "provider_estimate"
    assert payload["effective_output_tokens"] == 2_048


def test_capability_from_engine_prefers_explicit_runtime_metadata() -> None:
    class Engine:
        model_key = "local/model"
        n_ctx = 4_096
        model_max_output_tokens = 1_024
        tokenizer_name = "llama_cpp"
        config = None

    capability = capability_from_engine(Engine())

    assert capability.model_id == "local/model"
    assert capability.context_window == 4_096
    assert capability.max_output_tokens == 1_024
    assert capability.tokenizer == "llama_cpp"
    assert capability.source == "engine_metadata"


def test_capability_from_engine_marks_runtime_overrides() -> None:
    class Engine:
        model_key = "remote/model"
        config = None

    capability = capability_from_engine(
        Engine(),
        model_context_window=1_048_576,
        model_max_output_tokens=65_536,
    )

    assert capability.context_window == 1_048_576
    assert capability.max_output_tokens == 65_536
    assert capability.source == "runtime_options"


def test_capability_from_engine_distinguishes_n_ctx_from_registry_context() -> None:
    class Config:
        context_window = None

    class Engine:
        model_key = "local/model"
        config = Config()
        n_ctx = 4_096

    capability = capability_from_engine(Engine())

    assert capability.context_window == 4_096
    assert capability.source == "engine_metadata"


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
