from __future__ import annotations

from pathlib import Path

import numpy as np

from soca.core import AssistantRuntime, DefaultRuntimeToolRouter, RuntimeRoute
from soca.core.router_setup import build_runtime_tool_router
from soca.core.semantic_turn_router import build_semantic_turn_router
from soca.core.tool_routing import SemanticRouterConfig, ToolRouterConfig
from soca.knowledge import KnowledgeContextBuilder
from soca.llm.base import LLMResult
from soca.tools import KnowledgeSearchTool, ToolRuntime
from tests.fake_tools import ReadOnlyInspectTool
from tests.test_assistant_runtime import FakeKnowledgeSource, SpyLLM


class _SharedEmbedding:
    model_id = "fake:shared-turn-policy"

    def embed_documents(self, texts: tuple[str, ...]) -> np.ndarray:
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        return np.array(
            [1.0, 0.0] if "bayes" in text.lower() else [0.0, 1.0],
            dtype=np.float32,
        )


class _RouteFallbackLLM:
    def generate(self, user_msg: str, **kwargs: object) -> LLMResult:
        del user_msg, kwargs
        return LLMResult(
            '{"route":"retrieval_request","handler":null,"arguments":{},"sources":["knowledge"]}',
            "",
            0,
            0,
            0.0,
            0.0,
            0.0,
        )


def _router(tmp_path: Path, source: FakeKnowledgeSource | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    examples = tmp_path / "shared-turns.jsonl"
    examples.write_text(
        '{"id":"knowledge","query":"Ghi chú nói Bayes thế nào?",'
        '"route":"retrieval_request","handler":null,"sources":["knowledge"]}\n'
        '{"id":"smalltalk","query":"Xin chào",'
        '"route":"smalltalk","handler":null,"sources":[]}\n',
        encoding="utf-8",
    )
    router = build_semantic_turn_router(
        tool_runtime=ToolRuntime([KnowledgeSearchTool(source)]) if source is not None else ToolRuntime([]),
        config=SemanticRouterConfig(
            enabled=True,
            threshold=0.58,
            margin=0.04,
            examples_path=examples,
        ),
        embedding_model=_SharedEmbedding(),
    )
    assert router is not None
    return router


def test_chat_and_voice_use_the_same_retrieval_policy(tmp_path: Path) -> None:
    chat_source = FakeKnowledgeSource()
    voice_source = FakeKnowledgeSource()
    chat_llm = SpyLLM(text="Theo [K1], ghi chú nói về Bayes.")
    voice_llm = SpyLLM(text="Theo [K1], ghi chú nói về Bayes.")
    chat_runtime = AssistantRuntime(
        llm=chat_llm,
        tool_runtime=ToolRuntime([KnowledgeSearchTool(chat_source)]),
        knowledge_builder=KnowledgeContextBuilder(chat_source),
        tool_router=_router(tmp_path / "chat", chat_source),
    )
    voice_runtime = AssistantRuntime(
        llm=voice_llm,
        tool_runtime=ToolRuntime([KnowledgeSearchTool(voice_source)]),
        knowledge_builder=KnowledgeContextBuilder(voice_source),
        tool_router=_router(tmp_path / "voice", voice_source),
    )

    query = "Ghi chú nói Bayes thế nào?"
    chat_result = chat_runtime.run_text_turn(query, source="text")
    voice_result = voice_runtime.run_text_turn(query, source="asr")

    assert chat_result.route == voice_result.route == RuntimeRoute.KNOWLEDGE_LLM
    assert chat_result.trace is not None
    assert voice_result.trace is not None
    assert chat_result.trace.disposition == voice_result.trace.disposition == "retrieval_request"
    assert chat_result.trace.selected_sources == voice_result.trace.selected_sources == ("knowledge",)
    # Retrieval expands the candidate window before relevance filtering so the
    # user-visible top-k remains stable across chat and voice.
    assert chat_source.search_calls == [(query, 12)]
    assert voice_source.search_calls == [(query, 12)]
    assert "[K1]" in chat_llm.calls[0]["user_msg"]
    assert "[K1]" in voice_llm.calls[0]["user_msg"]


def test_asr_turn_does_not_use_the_removed_intent_gate(tmp_path: Path) -> None:
    source = FakeKnowledgeSource()
    llm = SpyLLM()
    runtime = AssistantRuntime(
        llm=llm,
        knowledge_builder=KnowledgeContextBuilder(source),
        tool_router=_router(tmp_path),
    )

    result = runtime.run_text_turn("Xin chào", source="asr")

    assert result.route == RuntimeRoute.FREE_CHAT
    assert source.search_calls == []


def test_router_setup_keeps_semantic_policy_enabled_for_voice(tmp_path: Path) -> None:
    (tmp_path / "turns.jsonl").write_text(
        '{"id":"knowledge","query":"Ghi chú nói Bayes thế nào?",'
        '"route":"retrieval_request","handler":null,"sources":["knowledge"]}\n',
        encoding="utf-8",
    )
    router = build_runtime_tool_router(
        llm=None,
        tool_runtime=ToolRuntime([]),
        deterministic=DefaultRuntimeToolRouter(enable_memory_search=False),
        config=ToolRouterConfig(
            mode="cascade",
            semantic=SemanticRouterConfig(
                enabled=True,
                threshold=0.58,
                margin=0.04,
                examples_path=(tmp_path / "turns.jsonl"),
            ),
        ),
        embedding_model=_SharedEmbedding(),
        voice=True,
    )

    assert router.select("Ghi chú nói Bayes thế nào?", knowledge_limit=3) is None
    assert router.last_tier == "none"
    assert router.last_decision.sources == ("knowledge",)


def test_cascade_uses_one_llm_attempt_after_semantic_uncertainty(tmp_path: Path) -> None:
    (tmp_path / "turns.jsonl").write_text(
        '{"id":"only","query":"một câu chắc chắn khác",'
        '"route":"smalltalk","sources":[]}\n',
        encoding="utf-8",
    )
    router = build_runtime_tool_router(
        llm=_RouteFallbackLLM(),
        tool_runtime=ToolRuntime([ReadOnlyInspectTool()]),
        deterministic=DefaultRuntimeToolRouter(enable_memory_search=False),
        config=ToolRouterConfig(
            mode="cascade",
            semantic=SemanticRouterConfig(
                enabled=True,
                threshold=0.99,
                margin=0.0,
                examples_path=tmp_path / "turns.jsonl",
            ),
        ),
        embedding_model=_SharedEmbedding(),
        voice=True,
    )

    assert router.select("Ghi chú Bayes của tôi ở đâu?", knowledge_limit=3) is None
    assert router.last_tier == "llm"
    assert router.last_decision.disposition == "retrieval_request"
