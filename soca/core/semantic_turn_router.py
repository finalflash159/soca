"""Open-set semantic capability policy shared by text and voice.

The router does not decide whether retrieved snippets are evidence and never
returns a synthetic tool for retrieval.  It only classifies a turn into an
execution disposition and, for retrieval, selects one or both local corpora.
Examples are data, not Vietnamese keyword rules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from soca.core.route_catalog import source_profile, validate_route_fields
from soca.core.tool_routing import SemanticRouterConfig, ToolRouterDecision, TurnDisposition
from soca.knowledge.retrievers.dense import EmbeddingModel
from soca.tools import ToolCall, ToolRuntime

_DISPOSITIONS: set[str] = {
    "direct_tool",
    "retrieval_request",
    "smalltalk",
    "out_of_scope",
    "unresolved",
}
_SOURCES = {"knowledge", "memory"}
MAX_EXAMPLES = 512
MAX_EXAMPLE_BYTES = 32_768
_PRODUCTION_EXAMPLE_SPLITS = frozenset({"train", "validation"})


@dataclass(frozen=True)
class SemanticTurnExample:
    disposition: TurnDisposition
    text: str
    sources: tuple[str, ...] = ()
    handler: str | None = None

    @property
    def route(self) -> TurnDisposition:
        return self.disposition

    @property
    def tool(self) -> str | None:
        """Compatibility alias for pre-P1 callers; the schema calls this handler."""
        return self.handler


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _load_examples(path: Path) -> tuple[SemanticTurnExample, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"semantic turn examples file not found: {path}")
    result: list[SemanticTurnExample] = []
    ids: set[str] = set()
    total_bytes = 0
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            total_bytes += len(raw_line)
            if total_bytes > MAX_EXAMPLE_BYTES:
                raise ValueError("semantic turn examples file is too large")
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            if not isinstance(payload, dict):
                raise ValueError(f"semantic turn example {line_number} must be an object")
            split = payload.get("split")
            if split is not None and split not in _PRODUCTION_EXAMPLE_SPLITS:
                continue
            example_id = payload.get("id")
            disposition = payload.get("route", payload.get("disposition"))
            text = _normalize_text(str(payload.get("text") or payload.get("query") or ""))
            sources = payload.get("sources", [])
            handler = payload.get("handler", payload.get("tool"))
            if (
                not isinstance(example_id, str)
                or not example_id
                or example_id in ids
                or disposition not in _DISPOSITIONS
                or not text
                or len(text) > 512
                or not isinstance(sources, list)
                or not set(sources) <= _SOURCES
                or (handler is not None and not isinstance(handler, str))
            ):
                raise ValueError(f"invalid semantic turn example {line_number}")
            try:
                _, normalized_sources, _ = validate_route_fields(
                    disposition,
                    handler=handler,
                    sources=sources,
                )
            except ValueError as exc:
                raise ValueError(f"invalid semantic turn example {line_number}: {exc}") from exc
            ids.add(example_id)
            result.append(
                SemanticTurnExample(
                    disposition=disposition,
                    text=text,
                    sources=normalized_sources,
                    handler=handler,
                )
            )
    if not result:
        raise ValueError("semantic turn examples file is empty")
    if len(result) > MAX_EXAMPLES:
        raise ValueError("too many semantic turn examples")
    return tuple(result)


class SemanticTurnRouter:
    def __init__(
        self,
        examples: tuple[SemanticTurnExample, ...],
        vectors: np.ndarray,
        tool_runtime: ToolRuntime,
        *,
        config: SemanticRouterConfig,
        embedding_model: EmbeddingModel,
    ) -> None:
        self._examples = examples
        self._vectors = vectors
        self._tool_runtime = tool_runtime
        self._config = config
        self._embedding_model = embedding_model
        self.last_tier = "none"
        self.last_decision = ToolRouterDecision()

    def select(self, text: str, *, knowledge_limit: int) -> ToolCall | None:
        del knowledge_limit
        normalized = _normalize_text(text)
        if not normalized:
            self.last_tier = "none"
            self.last_decision = ToolRouterDecision(reason="empty_input")
            return None
        try:
            vector = np.asarray(self._embedding_model.embed_query(normalized), dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if vector.ndim != 1 or norm <= 1e-12 or not np.isfinite(vector).all():
                raise ValueError("invalid query embedding")
            scores = self._vectors @ (vector / norm)
        except (OSError, RuntimeError, ValueError):
            self.last_tier = "none"
            self.last_decision = ToolRouterDecision(reason="embedding_unavailable")
            return None

        by_disposition: dict[str, float] = {}
        for example, raw_score in zip(self._examples, scores, strict=True):
            by_disposition[example.disposition] = max(
                by_disposition.get(example.disposition, float("-inf")), float(raw_score)
            )
        ranked = sorted(by_disposition.items(), key=lambda item: (-item[1], item[0]))
        top_name, top_score = ranked[0]
        runner_up = ranked[1][0] if len(ranked) > 1 else None
        margin = top_score - ranked[1][1] if len(ranked) > 1 else None
        score_map = {name: round(score, 6) for name, score in ranked}
        raw_source_scores = self._source_scores(scores)
        source_score_map = {
            source: round(score, 6) for source, score in raw_source_scores.items()
        }
        if top_score < self._config.threshold:
            self.last_tier = "none"
            self.last_decision = ToolRouterDecision(
                reason="below_threshold",
                scores=score_map,
                source_scores=source_score_map,
                selected_routes=(),
                runner_up=runner_up,
                margin=margin,
            )
            return None
        if margin is not None and margin < self._config.margin:
            self.last_tier = "none"
            self.last_decision = ToolRouterDecision(
                reason="ambiguous_margin",
                scores=score_map,
                source_scores=source_score_map,
                selected_routes=(),
                runner_up=runner_up,
                margin=margin,
            )
            return None

        disposition = top_name  # validated by _load_examples
        self.last_tier = "semantic"
        if disposition == "direct_tool":
            matching = [
                (example, float(score))
                for example, score in zip(self._examples, scores, strict=True)
                if example.disposition == disposition
            ]
            chosen, _ = max(matching, key=lambda pair: pair[1])
            tool = self._tool_runtime.get(chosen.handler or "")
            if tool is None or not tool.spec.enabled:
                self.last_tier = "none"
                self.last_decision = ToolRouterDecision(
                    reason="direct_tool_unavailable",
                    scores=score_map,
                    source_scores=source_score_map,
                    selected_routes=(),
                    runner_up=runner_up,
                    margin=margin,
                )
                return None
            call = ToolCall(chosen.handler or "", {})
            self.last_decision = ToolRouterDecision(
                call=call,
                reason="semantic_direct_tool",
                disposition="direct_tool",
                handler=chosen.handler,
                selected_routes=("direct_tool",),
                scores=score_map,
                source_scores=source_score_map,
                runner_up=runner_up,
                margin=margin,
            )
            return call
        if disposition == "retrieval_request":
            sources = self._select_sources(raw_source_scores)
            self.last_decision = ToolRouterDecision(
                reason="semantic_retrieval",
                disposition="retrieval_request",
                selected_routes=("retrieval_request",),
                sources=sources,
                source_profile=source_profile(sources),
                scores=score_map,
                source_scores=source_score_map,
                runner_up=runner_up,
                margin=margin,
            )
            return None
        self.last_decision = ToolRouterDecision(
            reason=f"semantic_{disposition}",
            disposition=disposition,  # type: ignore[arg-type]
            selected_routes=(cast(TurnDisposition, disposition),),
            scores=score_map,
            source_scores=source_score_map,
            runner_up=runner_up,
            margin=margin,
        )
        return None

    def _source_scores(self, scores: np.ndarray) -> dict[str, float]:
        by_source: dict[str, float] = {}
        for example, raw_score in zip(self._examples, scores, strict=True):
            if example.disposition != "retrieval_request":
                continue
            for source in example.sources:
                by_source[source] = max(by_source.get(source, float("-inf")), float(raw_score))
        return dict(sorted(by_source.items()))

    def _select_sources(self, by_source: dict[str, float]) -> tuple[str, ...]:
        if not by_source:
            return ()
        best_score = max(by_source.values())
        # Source selection is multi-label, not "every corpus above a global
        # floor".  A second corpus is selected only when it is genuinely close
        # to the best source; a labelled ambiguous example can therefore fan
        # out, while an unambiguous Bayes question stays knowledge-only.
        selected = tuple(
            source
            for source, score in sorted(by_source.items())
            if score >= self._config.threshold and best_score - score <= self._config.margin
        )
        # A confident retrieval request with no confident corpus is deliberately
        # unresolved.  The runtime asks for clarification; it must not search a
        # random corpus merely because retrieval was selected.
        return selected


def build_semantic_turn_router(
    *,
    tool_runtime: ToolRuntime,
    config: SemanticRouterConfig,
    embedding_model: EmbeddingModel | None,
) -> SemanticTurnRouter | None:
    if not config.enabled or config.examples_path is None or embedding_model is None:
        return None
    examples = _load_examples(config.examples_path)
    enabled: list[SemanticTurnExample] = []
    for example in examples:
        if example.disposition != "direct_tool":
            enabled.append(example)
            continue
        tool = tool_runtime.get(example.tool or "")
        if tool is not None and tool.spec.enabled:
            enabled.append(example)
    if not enabled:
        return None
    vectors = np.asarray(
        embedding_model.embed_documents(tuple(example.text for example in enabled)), dtype=np.float32
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(enabled) or not np.isfinite(vectors).all():
        raise ValueError("semantic turn embeddings have invalid shape")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("semantic turn embeddings have zero norm")
    normalized = np.ascontiguousarray(vectors / norms, dtype=np.float32)
    normalized.setflags(write=False)
    return SemanticTurnRouter(
        tuple(enabled), normalized, tool_runtime, config=config, embedding_model=embedding_model
    )


__all__ = ["SemanticTurnExample", "SemanticTurnRouter", "build_semantic_turn_router"]
