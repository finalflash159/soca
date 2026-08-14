from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from soca.core import AssistantRuntime, RuntimeOptions, RuntimeRoute
from soca.core.sufficient_context import (
    ContextSufficiencyAssessor,
    RetrievedContext,
    SufficiencyAssessmentError,
    SufficiencyDecision,
    SufficiencyPromptVariant,
    SufficiencyStatus,
    SufficientContextAutorater,
    parse_sufficiency_response,
    retrieved_contexts_from_tool_results,
)
from soca.core.workflow import ControlledWorkflowRunner, GoalContract, TerminalStatus
from soca.llm import LLMResult
from soca.llm.providers import RemoteLLMError
from soca.tools import ToolCall, ToolResult, ToolRuntime, ToolSpec, object_schema


@dataclass
class _StructuredLLM:
    responses: list[str]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def generate_structured(self, user_msg: str, **kwargs: Any) -> LLMResult:
        self.calls.append({"prompt": user_msg, **kwargs})
        return LLMResult(
            text=self.responses.pop(0),
            prompt=user_msg,
            n_prompt_tokens=20,
            n_completion_tokens=8,
            ttft_ms=12.0,
            total_latency_ms=20.0,
            tokens_per_second=400.0,
            provider_trace={"provider": "test", "model": "autorater"},
        )


@dataclass
class _KnowledgeTool:
    result: ToolResult

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="knowledge.search",
            description="Search bounded knowledge evidence.",
            input_schema=object_schema(
                properties={
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                required=["query"],
            ),
            workflow_capability="knowledge_search",
        )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        assert arguments["query"] == "Bayes"
        return self.result


def _knowledge_result() -> ToolResult:
    return ToolResult(
        "knowledge.search",
        True,
        "[K1] Bayes\nXác suất hậu nghiệm được cập nhật từ prior và likelihood.",
        data={
            "hits": [
                {
                    "path": "wiki/bayes.md",
                    "title": "Bayes",
                    "snippet": "Xác suất hậu nghiệm được cập nhật từ prior và likelihood.",
                }
            ]
        },
    )


def test_parse_sufficiency_response_accepts_only_typed_binary_output() -> None:
    decision = parse_sufficiency_response(
        json.dumps(
            {
                "sufficient": True,
                "confidence": 0.94,
                "reason_code": "answer_explicitly_supported",
            }
        )
    )

    assert decision.status is SufficiencyStatus.SUFFICIENT
    assert decision.confidence == pytest.approx(0.94)
    assert decision.reason_code == "answer_explicitly_supported"


@pytest.mark.parametrize(
    "payload",
    [
        {"sufficient": True, "confidence": 1.1, "reason_code": "bad"},
        {"sufficient": "yes", "confidence": 0.8, "reason_code": "bad"},
        {"sufficient": False, "confidence": 0.8, "reason_code": ""},
        {
            "sufficient": True,
            "confidence": 0.8,
            "reason_code": "supported",
            "reasoning": "private chain of thought",
        },
    ],
)
def test_parse_sufficiency_response_rejects_malformed_or_reasoning_fields(
    payload: dict[str, object],
) -> None:
    with pytest.raises(SufficiencyAssessmentError, match="invalid_output"):
        parse_sufficiency_response(json.dumps(payload))


def test_autorater_uses_one_structured_call_and_emits_no_reasoning_control_data() -> None:
    llm = _StructuredLLM(
        [
            json.dumps(
                {
                    "sufficient": False,
                    "confidence": 0.91,
                    "reason_code": "missing_requested_fact",
                }
            )
        ]
    )
    autorater = SufficientContextAutorater(
        llm,
        model_id="test/autorater",
        max_chars=2_000,
        prompt_variant=SufficiencyPromptVariant.STRICT_EXACT,
    )

    decision = autorater.assess(
        "Ai là tác giả của ghi chú?",
        (
            RetrievedContext(
                evidence_id="K1",
                text="Ghi chú chỉ mô tả định lý Bayes.",
                provenance={"path": "wiki/bayes.md"},
            ),
        ),
    )

    assert decision.status is SufficiencyStatus.INSUFFICIENT
    assert decision.model_id == "test/autorater"
    assert decision.evidence_ids == ("K1",)
    assert decision.provider_trace == {"provider": "test", "model": "autorater"}
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["schema_name"] == "soca_sufficient_context"
    assert call["inject_persona"] is False
    assert "reasoning" not in call["schema"]["properties"]
    assert "chain-of-thought" not in call["prompt"].lower()
    assert "closed-world exact entailment" in call["prompt"]
    assert "không sửa hộ câu hỏi" in call["prompt"].lower()
    assert "đảo vai" in call["prompt"].lower()


def test_paper_definition_prompt_allows_supported_inference_without_cot_output() -> None:
    llm = _StructuredLLM(
        [
            json.dumps(
                {
                    "sufficient": True,
                    "confidence": 0.9,
                    "reason_code": "supported_inference",
                }
            )
        ]
    )
    autorater = SufficientContextAutorater(
        llm,
        model_id="test/autorater",
        max_chars=3_000,
        prompt_variant=SufficiencyPromptVariant.PAPER_DEFINITION,
    )

    autorater.assess(
        "Lan sống ở đâu?",
        (RetrievedContext("K1", "Lan chuyển nhà đến Huế và vẫn ở đó.", {}),),
    )

    prompt = llm.calls[0]["prompt"].lower()
    assert "người đọc cẩn trọng" in prompt
    assert "suy luận" in prompt
    assert "lan chuyển nhà đến huế" in prompt
    assert "chain-of-thought" not in prompt
    assert "reasoning" not in llm.calls[0]["schema"]["properties"]


def test_balanced_prompt_contains_positive_and_hard_negative_examples() -> None:
    llm = _StructuredLLM(
        [
            json.dumps(
                {
                    "sufficient": False,
                    "confidence": 0.9,
                    "reason_code": "missing_requested_fact",
                }
            )
        ]
    )
    autorater = SufficientContextAutorater(
        llm,
        model_id="test/autorater",
        max_chars=4_000,
        prompt_variant=SufficiencyPromptVariant.BALANCED_EXAMPLES,
    )

    autorater.assess(
        "Ai viết tài liệu?",
        (RetrievedContext("K1", "Tài liệu nói về Bayes.", {}),),
    )

    prompt = llm.calls[0]["prompt"].lower()
    assert '"sufficient":true' in prompt
    assert '"sufficient":false' in prompt
    assert "đảo vai" in prompt


def test_autorater_rejects_unknown_prompt_variant() -> None:
    with pytest.raises(ValueError, match="prompt variant"):
        SufficientContextAutorater(
            _StructuredLLM([]),
            model_id="test/autorater",
            prompt_variant="invented",  # type: ignore[arg-type]
        )


def test_autorater_invalid_output_fails_closed_without_plain_generation_fallback() -> None:
    llm = _StructuredLLM(["not-json"])

    with pytest.raises(SufficiencyAssessmentError, match="invalid_output"):
        SufficientContextAutorater(llm, model_id="test/autorater").assess(
            "Câu hỏi",
            (RetrievedContext("K1", "Bằng chứng", {"path": "wiki/a.md"}),),
        )

    assert len(llm.calls) == 1


def test_autorater_preserves_typed_remote_failure_category() -> None:
    class RateLimitedLLM(_StructuredLLM):
        def generate_structured(self, user_msg: str, **kwargs: Any) -> LLMResult:
            del user_msg, kwargs
            raise RemoteLLMError("quota", category="rate_limit")

    with pytest.raises(SufficiencyAssessmentError, match="provider_rate_limit"):
        SufficientContextAutorater(
            RateLimitedLLM([]),
            model_id="test/autorater",
        ).assess(
            "Câu hỏi",
            (RetrievedContext("K1", "Bằng chứng", {}),),
        )


def test_controlled_workflow_stops_before_synthesis_when_context_is_insufficient() -> None:
    synth_calls: list[str] = []
    runner = ControlledWorkflowRunner(ToolRuntime([_KnowledgeTool(_knowledge_result())]))

    result = runner.run(
        GoalContract(goal_id="goal-1", objective="Ai viết ghi chú Bayes?"),
        explicit_call=ToolCall("knowledge.search", {"query": "Bayes"}),
        assess_context=lambda goal, actions, observations: SufficiencyDecision(
            status=SufficiencyStatus.INSUFFICIENT,
            confidence=0.97,
            reason_code="missing_requested_fact",
            evidence_ids=("K1",),
            model_id="test/autorater",
        ),
        synthesize=lambda goal, actions, observations: synth_calls.append(goal.objective)
        or "should not run",
    )

    assert result.terminal.status is TerminalStatus.INSUFFICIENT_EVIDENCE
    assert result.terminal.error_code == "insufficient_context"
    assert result.terminal.metadata["sufficiency"]["reason_code"] == "missing_requested_fact"
    assert result.state.evidence.status.value == "insufficient"
    assert synth_calls == []


def test_controlled_workflow_synthesizes_only_after_sufficient_context() -> None:
    order: list[str] = []
    runner = ControlledWorkflowRunner(ToolRuntime([_KnowledgeTool(_knowledge_result())]))

    result = runner.run(
        GoalContract(goal_id="goal-1", objective="Bayes cập nhật xác suất thế nào?"),
        explicit_call=ToolCall("knowledge.search", {"query": "Bayes"}),
        assess_context=lambda goal, actions, observations: order.append("assess")
        or SufficiencyDecision(
            status=SufficiencyStatus.SUFFICIENT,
            confidence=0.95,
            reason_code="answer_explicitly_supported",
            evidence_ids=("K1",),
            model_id="test/autorater",
        ),
        synthesize=lambda goal, actions, observations: order.append("synthesize")
        or "Bayes cập nhật posterior từ prior và likelihood.",
    )

    assert result.terminal.status is TerminalStatus.ACHIEVED
    assert order == ["assess", "synthesize"]
    assert result.state.evidence.status.value == "supported"


def test_controlled_workflow_fails_closed_when_autorater_is_unavailable() -> None:
    runner = ControlledWorkflowRunner(ToolRuntime([_KnowledgeTool(_knowledge_result())]))

    def fail(*args: object) -> SufficiencyDecision:
        del args
        raise SufficiencyAssessmentError("provider_unavailable")

    result = runner.run(
        GoalContract(goal_id="goal-1", objective="Bayes là gì?"),
        explicit_call=ToolCall("knowledge.search", {"query": "Bayes"}),
        assess_context=fail,
        synthesize=lambda goal, actions, observations: "should not run",
    )

    assert result.terminal.status is TerminalStatus.SYSTEM_FAILURE
    assert result.terminal.error_code == "sufficiency_assessment_failed"
    assert result.terminal.metadata["detail"] == "provider_unavailable"


def test_retrieved_contexts_use_selected_hit_text_and_stable_labels() -> None:
    contexts = retrieved_contexts_from_tool_results((_knowledge_result(),))

    assert contexts == (
        RetrievedContext(
            evidence_id="K1",
            text="Xác suất hậu nghiệm được cập nhật từ prior và likelihood.",
            provenance={"path": "wiki/bayes.md", "title": "Bayes"},
        ),
    )


@dataclass
class _StaticAssessor(ContextSufficiencyAssessor):
    decision: SufficiencyDecision
    calls: list[tuple[str, tuple[RetrievedContext, ...]]] = field(default_factory=list)

    def assess(
        self,
        question: str,
        contexts: tuple[RetrievedContext, ...],
    ) -> SufficiencyDecision:
        self.calls.append((question, contexts))
        return self.decision


def test_assistant_runtime_returns_typed_abstention_without_calling_generator() -> None:
    assessor = _StaticAssessor(
        SufficiencyDecision(
            status=SufficiencyStatus.INSUFFICIENT,
            confidence=0.96,
            reason_code="missing_requested_fact",
            evidence_ids=("K1",),
            model_id="test/autorater",
        )
    )
    runtime = AssistantRuntime(
        tool_runtime=ToolRuntime([_KnowledgeTool(_knowledge_result())]),
        sufficiency_assessor=assessor,
        options=RuntimeOptions(require_sufficient_context=True),
    )

    result = runtime.run_text_turn("wiki: Bayes")

    assert result.blocked is False
    assert result.route is RuntimeRoute.KNOWLEDGE_LLM
    assert result.citations == ()
    assert "chưa đủ bằng chứng" in result.response_text.lower()
    assert result.trace is not None
    assert result.trace.used_llm is True
    assert result.trace.answer_policy == "abstain"
    assert result.trace.workflow_status == "insufficient_evidence"
    assert result.trace.workflow_error_code == "insufficient_context"
    assert result.trace.sufficiency_decision.status is SufficiencyStatus.INSUFFICIENT
    assert len(assessor.calls) == 1


def test_assistant_runtime_requires_configured_autorater_when_gate_is_enabled() -> None:
    runtime = AssistantRuntime(
        tool_runtime=ToolRuntime([_KnowledgeTool(_knowledge_result())]),
        options=RuntimeOptions(require_sufficient_context=True),
    )

    result = runtime.run_text_turn("wiki: Bayes")

    assert result.blocked is True
    assert result.trace is not None
    assert result.trace.workflow_status == "system_failure"
    assert result.trace.workflow_error_code == "sufficiency_assessment_failed"
