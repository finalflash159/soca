from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from soca.core.tool_routing import SemanticRouterConfig, ToolRouterDecision
from soca.knowledge.retrievers.dense import EmbeddingModel
from soca.tools import ToolCall, ToolRuntime
from soca.tools.base import validate_arguments

ArgumentPolicy = Literal["none", "raw_query"]
MAX_EXAMPLE_BYTES = 32_768
MAX_EXAMPLES = 512


@dataclass(frozen=True)
class SemanticRouteExample:
    tool: str
    text: str
    argument_policy: ArgumentPolicy


@dataclass(frozen=True)
class _EmbeddedExample:
    example: SemanticRouteExample
    vector: np.ndarray


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _load_examples(path: Path) -> tuple[SemanticRouteExample, ...]:
    resolved = path.expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"semantic examples file not found: {resolved}")
    examples: list[SemanticRouteExample] = []
    ids: set[str] = set()
    total_bytes = 0
    with resolved.open("rb") as raw_file:
        for line_number, raw_line in enumerate(raw_file, start=1):
            total_bytes += len(raw_line)
            if total_bytes > MAX_EXAMPLE_BYTES:
                raise ValueError("semantic examples file is too large")
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"semantic example {line_number} must be an object")
            example_id = payload.get("id")
            tool = payload.get("tool")
            text = _normalize_text(str(payload.get("text", "")))
            policy = payload.get("argument_policy", "none")
            if (
                not isinstance(example_id, str)
                or not example_id.strip()
                or example_id in ids
                or not isinstance(tool, str)
                or not tool.strip()
                or len(text) > 512
                or not text
                or policy not in {"none", "raw_query"}
            ):
                raise ValueError(f"invalid semantic example {line_number}")
            ids.add(example_id)
            examples.append(
                SemanticRouteExample(tool=tool, text=text, argument_policy=policy)
            )
            if len(examples) > MAX_EXAMPLES:
                raise ValueError("too many semantic examples")
    return tuple(examples)


class SemanticToolRouter:
    def __init__(
        self,
        examples: tuple[SemanticRouteExample, ...],
        vectors: np.ndarray,
        tool_runtime: ToolRuntime,
        *,
        config: SemanticRouterConfig,
        embedding_model: EmbeddingModel,
    ) -> None:
        self._examples = examples
        self._vectors = np.asarray(vectors, dtype=np.float32)
        self._tool_runtime = tool_runtime
        self._config = config
        self._embedding_model = embedding_model
        self.last_tier = "none"
        self.last_decision = ToolRouterDecision()

    def select(self, text: str, *, knowledge_limit: int) -> ToolCall | None:
        normalized = _normalize_text(text)
        if not normalized or not self._examples:
            self.last_tier = "none"
            self.last_decision = ToolRouterDecision(reason="empty_or_no_examples")
            return None
        try:
            query_vector = np.asarray(
                self._embedding_model.embed_query(normalized), dtype=np.float32
            )
            query_norm = float(np.linalg.norm(query_vector))
            if query_vector.ndim != 1 or query_norm <= 1e-12 or not np.isfinite(query_norm):
                raise ValueError("query embedding is invalid")
            query_vector = query_vector / query_norm
            scores = self._vectors @ query_vector
        except (OSError, RuntimeError, ValueError):
            self.last_tier = "none"
            self.last_decision = ToolRouterDecision(reason="embedding_unavailable")
            return None

        best_by_tool: dict[str, float] = {}
        policies: dict[str, ArgumentPolicy] = {}
        for example, score in zip(self._examples, scores, strict=True):
            value = float(score)
            if value > best_by_tool.get(example.tool, float("-inf")):
                best_by_tool[example.tool] = value
                policies[example.tool] = example.argument_policy
        ranked = sorted(best_by_tool.items(), key=lambda item: (-item[1], item[0]))
        if not ranked or ranked[0][1] < self._config.threshold:
            self.last_tier = "none"
            self.last_decision = ToolRouterDecision(reason="below_threshold")
            return None
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < self._config.margin:
            self.last_tier = "none"
            self.last_decision = ToolRouterDecision(reason="ambiguous_margin")
            return None

        tool_name, _ = ranked[0]
        if tool_name == "none":
            self.last_tier = "semantic"
            self.last_decision = ToolRouterDecision(reason="semantic_none")
            return None
        tool = self._tool_runtime.get(tool_name)
        if tool is None or not tool.spec.enabled:
            self.last_tier = "none"
            self.last_decision = ToolRouterDecision(reason="tool_unavailable")
            return None
        policy = policies[tool_name]
        if policy == "none":
            arguments: dict[str, object] = {}
        elif policy == "raw_query" and tool_name in {"knowledge.search", "memory.search"}:
            arguments = {"query": normalized, "limit": knowledge_limit}
        else:
            self.last_tier = "none"
            self.last_decision = ToolRouterDecision(reason="argument_policy_invalid")
            return None
        if validate_arguments(tool.spec.input_schema, arguments):
            self.last_tier = "none"
            self.last_decision = ToolRouterDecision(reason="invalid_arguments")
            return None
        self.last_tier = "semantic"
        call = ToolCall(tool_name, arguments)
        self.last_decision = ToolRouterDecision(call=call, reason="semantic_match")
        return call


def build_semantic_router(
    *,
    tool_runtime: ToolRuntime,
    config: SemanticRouterConfig,
    embedding_model: EmbeddingModel | None,
) -> SemanticToolRouter | None:
    if not config.enabled or embedding_model is None or config.examples_path is None:
        return None
    loaded_examples = _load_examples(config.examples_path)
    examples: list[SemanticRouteExample] = []
    for example in loaded_examples:
        if example.tool == "none":
            if example.argument_policy != "none":
                raise ValueError("none route only accepts the none argument policy")
            examples.append(example)
            continue
        tool = tool_runtime.get(example.tool)
        if tool is None or not tool.spec.enabled:
            # A disabled optional tool must not disable unrelated semantic routes.
            continue
        if example.argument_policy == "raw_query" and example.tool not in {
            "knowledge.search",
            "memory.search",
        }:
            raise ValueError("raw_query policy is only allowed for search tools")
        if example.argument_policy == "none" and tool.spec.input_schema.get("required"):
            raise ValueError(f"none policy requires no required arguments: {example.tool}")
        examples.append(example)
    if not examples:
        return None
    example_tuple = tuple(examples)
    vectors = embedding_model.embed_documents(tuple(example.text for example in example_tuple))
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(example_tuple):
        raise ValueError("semantic example embeddings have an invalid shape")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12) or not np.isfinite(matrix).all():
        raise ValueError("semantic example embeddings are invalid")
    normalized = np.ascontiguousarray(matrix / norms, dtype=np.float32)
    normalized.setflags(write=False)
    return SemanticToolRouter(
        example_tuple,
        normalized,
        tool_runtime,
        config=config,
        embedding_model=embedding_model,
    )


__all__ = [
    "SemanticRouteExample",
    "SemanticToolRouter",
    "build_semantic_router",
]
