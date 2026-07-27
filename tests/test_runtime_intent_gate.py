from __future__ import annotations

from soca.core import AssistantRuntime, RuntimeOptions, RuntimeRoute
from soca.knowledge import KnowledgeContextBuilder, KnowledgeDocument, KnowledgeHit
from soca.knowledge.intent_gate import IntentDecision
from tests.test_assistant_runtime import FakeKnowledgeSource, SpyLLM


class SpyGate:
    def __init__(self, use: bool) -> None:
        self.use = use
        self.calls: list[tuple[str, int]] = []

    def evaluate(self, query: str, *, limit: int) -> IntentDecision:
        self.calls.append((query, limit))
        hit = KnowledgeHit(KnowledgeDocument("wiki/a", "wiki/a.md", "A", "fact"), 0.8, "fact")
        return IntentDecision(self.use, "test", 0.8, (hit,) if self.use else ())


def test_asr_intent_false_does_not_build_context() -> None:
    gate = SpyGate(False)
    llm = SpyLLM()
    runtime = AssistantRuntime(
        llm=llm,
        knowledge_builder=KnowledgeContextBuilder(FakeKnowledgeSource()),
        knowledge_intent_gate=gate,
        options=RuntimeOptions(voice_knowledge_mode="intent"),
    )
    result = runtime.run_text_turn("xin chào", source="asr")
    assert result.route == RuntimeRoute.FREE_CHAT
    assert gate.calls == [("xin chào", 3)]
    assert "Local knowledge" not in llm.calls[0]["user_msg"]


def test_asr_intent_true_builds_from_gate_hits() -> None:
    gate = SpyGate(True)
    llm = SpyLLM()
    runtime = AssistantRuntime(
        llm=llm,
        knowledge_builder=KnowledgeContextBuilder(FakeKnowledgeSource()),
        knowledge_intent_gate=gate,
        options=RuntimeOptions(voice_knowledge_mode="intent"),
    )
    result = runtime.run_text_turn("protein là gì", source="asr")
    assert result.route == RuntimeRoute.KNOWLEDGE_LLM
    assert result.citations[0].path == "wiki/a.md"
    assert gate.calls == [("protein là gì", 3)]


def test_text_source_does_not_call_intent_gate() -> None:
    gate = SpyGate(True)
    runtime = AssistantRuntime(
        llm=SpyLLM(),
        knowledge_builder=KnowledgeContextBuilder(FakeKnowledgeSource()),
        knowledge_intent_gate=gate,
        options=RuntimeOptions(voice_knowledge_mode="intent"),
    )
    result = runtime.run_text_turn("protein là gì", source="text")
    assert result.route == RuntimeRoute.FREE_CHAT
    assert gate.calls == []


def test_asr_always_uses_normal_builder() -> None:
    gate = SpyGate(False)
    source = FakeKnowledgeSource()
    runtime = AssistantRuntime(
        llm=SpyLLM(),
        knowledge_builder=KnowledgeContextBuilder(source),
        knowledge_intent_gate=gate,
        options=RuntimeOptions(voice_knowledge_mode="always"),
    )
    result = runtime.run_text_turn("protein", source="asr")
    assert result.route == RuntimeRoute.KNOWLEDGE_LLM
    assert source.search_calls == [("protein", 4)]
    assert gate.calls == []
