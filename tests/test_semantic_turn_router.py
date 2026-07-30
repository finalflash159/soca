from __future__ import annotations

from pathlib import Path

import numpy as np

from soca.core.semantic_turn_router import (
    SemanticTurnExample,
    SemanticTurnRouter,
    build_semantic_turn_router,
)
from soca.core.tool_routing import SemanticRouterConfig
from soca.tools import ToolRuntime
from tests.fake_tools import ReadOnlyCatalogTool


class _Embedding:
    model_id = "fake:turn-policy"

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        text = text.lower()
        if "thời tiết" in text or "nhiệt độ" in text:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if "kho ghi chú" in text or "liên kết" in text:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if "onnx" in text:
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.2, 0.8], dtype=np.float32)

    def embed_documents(self, texts: tuple[str, ...]) -> np.ndarray:
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


def test_semantic_policy_keeps_weather_out_of_scope_and_selects_both_sources(tmp_path: Path) -> None:
    examples = tmp_path / "turns.jsonl"
    examples.write_text(
        "\n".join(
            [
                '{"id":"catalog","text":"kho ghi chú có gì","disposition":"direct_tool","sources":[],"tool":"knowledge.catalog"}',
                '{"id":"both","text":"tôi đã ghi gì về ONNX","disposition":"retrieval_request","sources":["knowledge","memory"]}',
                '{"id":"weather","text":"thời tiết hiện tại","disposition":"out_of_scope","sources":[]}',
            ]
        ) + "\n",
        encoding="utf-8",
    )
    router = build_semantic_turn_router(
        tool_runtime=ToolRuntime([ReadOnlyCatalogTool()]),
        config=SemanticRouterConfig(enabled=True, threshold=0.7, margin=0.05, examples_path=examples),
        embedding_model=_Embedding(),
    )
    assert router is not None

    assert router.select("Nhiệt độ hiện tại bao nhiêu?", knowledge_limit=3) is None
    assert router.last_decision.disposition == "out_of_scope"
    assert router.last_decision.call is None

    assert router.select("Tôi đã ghi gì về ONNX Runtime?", knowledge_limit=3) is None
    assert router.last_decision.disposition == "retrieval_request"
    assert router.last_decision.sources == ("knowledge", "memory")
    assert set(router.last_decision.source_scores) == {"knowledge", "memory"}


def test_semantic_policy_uses_only_allowlisted_direct_tool(tmp_path: Path) -> None:
    examples = tmp_path / "turns.jsonl"
    examples.write_text(
        '{"id":"catalog","text":"kho ghi chú có gì","disposition":"direct_tool","sources":[],"tool":"knowledge.catalog"}\n',
        encoding="utf-8",
    )
    router = build_semantic_turn_router(
        tool_runtime=ToolRuntime([ReadOnlyCatalogTool()]),
        config=SemanticRouterConfig(enabled=True, threshold=0.7, margin=0.0, examples_path=examples),
        embedding_model=_Embedding(),
    )
    assert router is not None
    call = router.select("Kho ghi chú hiện có gì?", knowledge_limit=3)
    assert call is not None
    assert call.name == "knowledge.catalog"


def test_source_threshold_uses_raw_score_not_rounded_telemetry() -> None:
    examples = (
        SemanticTurnExample("retrieval_request", "knowledge", ("knowledge",)),
        SemanticTurnExample("retrieval_request", "memory", ("memory",)),
    )
    near_threshold = 0.5799996
    vectors = np.asarray(
        [[1.0, 0.0], [near_threshold, np.sqrt(1.0 - near_threshold**2)]],
        dtype=np.float32,
    )

    class _Query:
        model_id = "fake:threshold"

        def embed_query(self, text: str) -> np.ndarray:
            del text
            return np.array([1.0, 0.0], dtype=np.float32)

        def embed_documents(self, texts: tuple[str, ...]) -> np.ndarray:
            del texts
            return vectors

    router = SemanticTurnRouter(
        examples,
        vectors,
        ToolRuntime([]),
        config=SemanticRouterConfig(enabled=False, threshold=0.58, margin=0.0),
        embedding_model=_Query(),
    )

    router.select("memory", knowledge_limit=3)

    assert router.last_decision.sources == ("knowledge",)
    assert router.last_decision.source_scores["memory"] == 0.58


def test_production_loader_excludes_sealed_test_split(tmp_path: Path) -> None:
    examples = tmp_path / "split.jsonl"
    examples.write_text(
        '{"id":"train","split":"train","query":"Bayes",'
        '"route":"retrieval_request","sources":["knowledge"]}\n'
        '{"id":"test","split":"test","query":"weather",'
        '"route":"out_of_scope","sources":[]}\n',
        encoding="utf-8",
    )
    router = build_semantic_turn_router(
        tool_runtime=ToolRuntime([]),
        config=SemanticRouterConfig(
            enabled=True,
            threshold=0.0,
            margin=0.0,
            examples_path=examples,
        ),
        embedding_model=_Embedding(),
    )

    assert router is not None
    assert len(router._examples) == 1


def test_semantic_router_delegates_uncertain_direct_tool_to_next_tier(tmp_path: Path) -> None:
    examples = tmp_path / "turns.jsonl"
    examples.write_text(
        "\n".join(
            [
                '{"id":"catalog","text":"kho ghi chú có gì",'
                '"disposition":"direct_tool","sources":[],"tool":"knowledge.catalog"}',
                '{"id":"search","text":"tìm nội dung trong ghi chú",'
                '"disposition":"retrieval_request","sources":["knowledge"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    class _AmbiguousEmbedding:
        model_id = "fake:ambiguous"

        def embed_documents(self, texts: tuple[str, ...]) -> np.ndarray:
            del texts
            return np.asarray([[0.8, 0.6], [0.79, 0.613]], dtype=np.float32)

        def embed_query(self, text: str) -> np.ndarray:
            del text
            return np.asarray([1.0, 0.0], dtype=np.float32)

    router = build_semantic_turn_router(
        tool_runtime=ToolRuntime([ReadOnlyCatalogTool()]),
        config=SemanticRouterConfig(
            enabled=True,
            threshold=0.0,
            margin=0.0,
            direct_tool_threshold=0.85,
            direct_tool_retrieval_margin=0.0,
            examples_path=examples,
        ),
        embedding_model=_AmbiguousEmbedding(),
    )
    assert router is not None

    assert router.select("Kho ghi chú có gì?", knowledge_limit=3) is None
    assert router.last_tier == "none"
    assert router.last_decision.reason == "semantic_direct_tool_uncertain"
    assert router.last_decision.disposition == "unresolved"
