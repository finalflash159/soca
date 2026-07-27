from __future__ import annotations

from pathlib import Path

import numpy as np

from soca.core.semantic_tool_router import (
    SemanticRouteExample,
    SemanticToolRouter,
    build_semantic_router,
)
from soca.core.tool_routing import SemanticRouterConfig
from soca.tools import LocalTimeTool, ToolCall, ToolResult, ToolRuntime, ToolSpec, object_schema


class _SearchTool:
    def __init__(self, name: str) -> None:
        self._spec = ToolSpec(
            name=name,
            description="test search",
            input_schema=object_schema(
                properties={
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                required=["query"],
            ),
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def run(self, arguments: dict[str, object]) -> ToolResult:
        return ToolResult(self.spec.name, True, "ok")


class _FakeEmbedding:
    model_id = "fake:router"

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        lowered = text.lower()
        if any(word in lowered for word in ("chào", "cảm ơn", "trò chuyện", "none")):
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if any(word in lowered for word in ("giờ", "time")):
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if any(word in lowered for word in ("memory", "cá nhân")):
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.8, 0.2], dtype=np.float32)

    def embed_documents(self, texts: tuple[str, ...]) -> np.ndarray:
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


def _runtime() -> ToolRuntime:
    return ToolRuntime([LocalTimeTool(), _SearchTool("knowledge.search"), _SearchTool("memory.search")])


def test_semantic_router_supports_explicit_none_and_memory_search() -> None:
    examples = (
        SemanticRouteExample("local_time.now", "bây giờ là mấy giờ", "none"),
        SemanticRouteExample("memory.search", "tìm trong memory cá nhân", "raw_query"),
        SemanticRouteExample("none", "xin chào, chỉ trò chuyện", "none"),
    )
    vectors = _FakeEmbedding().embed_documents(tuple(item.text for item in examples))
    router = SemanticToolRouter(
        examples,
        vectors / np.linalg.norm(vectors, axis=1, keepdims=True),
        _runtime(),
        config=SemanticRouterConfig(enabled=True, threshold=0.7, margin=0.05, examples_path=Path("examples")),
        embedding_model=_FakeEmbedding(),
    )

    call = router.select("tìm trong memory cá nhân của tôi", knowledge_limit=3)
    assert call == ToolCall("memory.search", {"query": "tìm trong memory cá nhân của tôi", "limit": 3})
    assert router.last_decision.reason == "semantic_match"

    assert router.select("xin chào", knowledge_limit=3) is None
    assert router.last_tier == "semantic"
    assert router.last_decision.reason == "semantic_none"


def test_semantic_example_loader_accepts_none_route(tmp_path: Path) -> None:
    examples_path = tmp_path / "examples.jsonl"
    examples_path.write_text(
        '{"id":"none","text":"xin chào","tool":"none","argument_policy":"none"}\n',
        encoding="utf-8",
    )
    router = build_semantic_router(
        tool_runtime=_runtime(),
        config=SemanticRouterConfig(
            enabled=True,
            threshold=0.5,
            margin=0.01,
            examples_path=examples_path,
        ),
        embedding_model=_FakeEmbedding(),
    )
    assert router is not None
    assert router.select("xin chào", knowledge_limit=3) is None
    assert router.last_decision.reason == "semantic_none"
