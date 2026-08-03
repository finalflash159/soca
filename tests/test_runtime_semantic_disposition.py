from __future__ import annotations

from soca.core import AssistantRuntime, RuntimeOptions, RuntimeRoute
from soca.core.tool_routing import ToolRouterDecision
from soca.knowledge import KnowledgeContextBuilder
from soca.memory import MemoryContextBuilder
from soca.tools import KnowledgeSearchTool, ToolCall, ToolRuntime
from tests.test_assistant_runtime import FakeKnowledgeSource, FakeRetrievedMemory, SpyLLM


class _Router:
    def __init__(self, decision: ToolRouterDecision) -> None:
        self.last_tier = "semantic"
        self.last_decision = decision

    def select(self, text: str, *, knowledge_limit: int) -> ToolCall | None:
        del text, knowledge_limit
        return self.last_decision.call


def test_out_of_scope_label_does_not_terminally_block_llm() -> None:
    llm = SpyLLM()
    runtime = AssistantRuntime(
        llm=llm,
        tool_router=_Router(
            ToolRouterDecision(reason="semantic_out_of_scope", disposition="out_of_scope")
        ),
    )
    result = runtime.run_text_turn("Thời tiết hiện tại ở Hà Nội thế nào?")
    assert result.route == RuntimeRoute.FREE_CHAT
    assert result.trace is not None
    assert result.trace.tool_calls == ()
    assert llm.calls


def test_semantic_knowledge_request_builds_context_then_calls_llm() -> None:
    source = FakeKnowledgeSource()
    llm = SpyLLM(text="Theo [K1], ghi chú nói về Bayes.")
    runtime = AssistantRuntime(
        llm=llm,
        tool_runtime=ToolRuntime([KnowledgeSearchTool(source)]),
        knowledge_builder=KnowledgeContextBuilder(source),
        tool_router=_Router(
            ToolRouterDecision(
                reason="semantic_retrieval",
                disposition="retrieval_request",
                sources=("knowledge",),
            )
        ),
        options=RuntimeOptions(turn_workflow="shadow"),
    )
    result = runtime.run_text_turn("Ghi chú nói Bayes thế nào?")
    assert result.route == RuntimeRoute.KNOWLEDGE_LLM
    assert result.trace is not None
    assert result.trace.selected_sources == ("knowledge",)
    assert result.trace.tool_calls[0].name == "knowledge.search"
    assert result.trace.evidence_decisions[0].status == "weak"
    assert result.trace.answer_validation.status == "valid"
    assert result.trace.answer_policy == "grounded"
    assert "[K1]" in llm.calls[0]["user_msg"]


def test_joint_knowledge_memory_request_discloses_unreconciled_sources() -> None:
    source = FakeKnowledgeSource()
    llm = SpyLLM(
        text=(
            "Knowledge nói protein hỗ trợ cơ bắp [K1], còn memory ghi chọn "
            "TTS local vì riêng tư [M1]; hai nguồn nói về hai nội dung khác nhau."
        )
    )
    runtime = AssistantRuntime(
        llm=llm,
        knowledge_builder=KnowledgeContextBuilder(source),
        memory_builder=MemoryContextBuilder(long_term=FakeRetrievedMemory()),
        tool_router=_Router(
            ToolRouterDecision(
                reason="semantic_retrieval",
                disposition="retrieval_request",
                sources=("knowledge", "memory"),
            )
        ),
        options=RuntimeOptions(turn_workflow="shadow"),
    )

    result = runtime.run_text_turn(
        "So sánh knowledge và memory của tôi",
        metadata={"evidence_relation": "conflicting"},
    )

    assert result.route == RuntimeRoute.KNOWLEDGE_LLM
    assert result.trace is not None
    assert result.trace.evidence_bundle.status == "conflicting"
    assert result.trace.answer_policy == "conflict_disclosure"
    assert result.trace.citation_count == 2
    assert result.trace.memory_access_plan.archive_mode == "semantic"
    assert result.trace.answer_validation.status == "valid"
    assert "Không so độ lớn score giữa hai nguồn" in llm.calls[0]["user_msg"]
    assert "[K1]" in llm.calls[0]["user_msg"]
    assert "[M1]" in llm.calls[0]["user_msg"]
