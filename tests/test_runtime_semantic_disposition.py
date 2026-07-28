from __future__ import annotations

from soca.core import AssistantRuntime, RuntimeRoute
from soca.core.tool_routing import ToolRouterDecision
from soca.knowledge import KnowledgeContextBuilder
from soca.tools import ToolCall
from tests.test_assistant_runtime import FakeKnowledgeSource, SpyLLM


class _Router:
    def __init__(self, decision: ToolRouterDecision) -> None:
        self.last_tier = "semantic"
        self.last_decision = decision

    def select(self, text: str, *, knowledge_limit: int) -> ToolCall | None:
        del text, knowledge_limit
        return self.last_decision.call


def test_out_of_scope_does_not_call_llm_or_direct_tool() -> None:
    llm = SpyLLM()
    runtime = AssistantRuntime(
        llm=llm,
        tool_router=_Router(
            ToolRouterDecision(reason="semantic_out_of_scope", disposition="out_of_scope")
        ),
    )
    result = runtime.run_text_turn("Thời tiết hiện tại ở Hà Nội thế nào?")
    assert result.route == RuntimeRoute.OUT_OF_SCOPE
    assert result.trace is not None
    assert result.trace.tool_calls == ()
    assert llm.calls == []


def test_semantic_knowledge_request_builds_context_then_calls_llm() -> None:
    source = FakeKnowledgeSource()
    llm = SpyLLM()
    runtime = AssistantRuntime(
        llm=llm,
        knowledge_builder=KnowledgeContextBuilder(source),
        tool_router=_Router(
            ToolRouterDecision(
                reason="semantic_retrieval",
                disposition="retrieval_request",
                sources=("knowledge",),
            )
        ),
    )
    result = runtime.run_text_turn("Ghi chú nói Bayes thế nào?")
    assert result.route == RuntimeRoute.KNOWLEDGE_LLM
    assert result.trace is not None
    assert result.trace.selected_sources == ("knowledge",)
    assert result.trace.tool_calls == ()
    assert result.trace.evidence_decisions[0].status == "supported"
    assert "[K1]" in llm.calls[0]["user_msg"]
