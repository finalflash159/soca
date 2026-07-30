from __future__ import annotations

from pathlib import Path

import numpy as np

from soca.core.semantic_turn_router import (
    SemanticTurnExample,
    SemanticTurnRouter,
    build_semantic_turn_router,
)
from soca.core.tool_routing import SemanticRouterConfig
from soca.tools import LocalTimeTool, ToolRuntime


class _Embedding:
    model_id = "fake:turn-policy"

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        text = text.lower()
        if "thời tiết" in text or "nhiệt độ" in text:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if "mấy giờ" in text or "giờ địa" in text:
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
                '{"id":"time","text":"bây giờ là mấy giờ","disposition":"direct_tool","sources":[],"tool":"local_time.now"}',
                '{"id":"both","text":"tôi đã ghi gì về ONNX","disposition":"retrieval_request","sources":["knowledge","memory"]}',
                '{"id":"weather","text":"thời tiết hiện tại","disposition":"out_of_scope","sources":[]}',
            ]
        ) + "\n",
        encoding="utf-8",
    )
    router = build_semantic_turn_router(
        tool_runtime=ToolRuntime([LocalTimeTool()]),
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
        '{"id":"time","text":"bây giờ là mấy giờ","disposition":"direct_tool","sources":[],"tool":"local_time.now"}\n',
        encoding="utf-8",
    )
    router = build_semantic_turn_router(
        tool_runtime=ToolRuntime([LocalTimeTool()]),
        config=SemanticRouterConfig(enabled=True, threshold=0.7, margin=0.0, examples_path=examples),
        embedding_model=_Embedding(),
    )
    assert router is not None
    call = router.select("Bây giờ là mấy giờ?", knowledge_limit=3)
    assert call is not None
    assert call.name == "local_time.now"


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
