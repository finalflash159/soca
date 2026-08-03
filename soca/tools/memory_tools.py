from __future__ import annotations

from typing import Any

from soca.memory.context import MemoryContextBuilder
from soca.tools.base import (
    InvalidToolInput,
    SideEffectLevel,
    ToolResult,
    ToolSpec,
    object_schema,
)


def _hit_payload(hit: object) -> dict[str, Any]:
    document = getattr(hit, "document", None)
    payload: dict[str, Any] = {
        "path": str(getattr(document, "path", "")),
        "title": str(getattr(document, "title", "")),
        "snippet": str(getattr(hit, "snippet", "")),
    }
    score = getattr(hit, "score", None)
    if score is not None:
        payload["score"] = getattr(score, "total", score)
    for field in ("line_start", "line_end"):
        value = getattr(hit, field, None)
        if value is not None:
            payload[field] = value
    for field in ("retrieval_backend", "sparse_score", "dense_score", "fusion_score"):
        value = getattr(hit, field, None)
        if value is not None:
            payload[field] = value
    return payload


class MemorySearchTool:
    def __init__(self, builder: MemoryContextBuilder, max_limit: int = 5) -> None:
        self.builder = builder
        self.max_limit = max_limit

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="memory.search",
            description="Search the user's local long-term memory notes without exposing the knowledge wiki.",
            input_schema=object_schema(
                properties={
                    "query": {"type": "string", "description": "Memory search query."},
                    "limit": {"type": "integer", "description": "Maximum number of hits."},
                },
                required=["query"],
            ),
            side_effect=SideEffectLevel.READ_ONLY,
            workflow_capability="memory_search",
        )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        query = str(arguments["query"]).strip()
        if not query:
            raise InvalidToolInput("empty_query")
        limit = max(1, min(int(arguments.get("limit") or self.max_limit), self.max_limit))
        context = self.builder.build(query, include_archive=True)
        hits = tuple(context.hits)[:limit]
        data_hits = [_hit_payload(hit) for hit in hits]
        if not hits:
            return ToolResult(
                name=self.spec.name,
                ok=True,
                content="Mình chưa tìm thấy ghi chú phù hợp trong memory.",
                data={
                    "hits": [],
                    "mode": context.mode,
                    "evidence_status": context.evidence_status,
                    "evidence_reason": context.evidence_reason,
                    "rejected_hit_count": context.rejected_hit_count,
                    "top_relevance": context.top_relevance,
                    "relevance_margin": context.relevance_margin,
                    "score_separation": context.score_separation,
                    "query_coverage": context.query_coverage,
                    "sparse_top_score": context.sparse_top_score,
                    "dense_top_score": context.dense_top_score,
                    "retrieval_state": context.retrieval_state,
                    "retrieval_reason": context.retrieval_reason,
                },
            )
        lines = [
            f"[M{index}] {item['title']} ({item['path']})\n{item['snippet']}"
            for index, item in enumerate(data_hits, start=1)
        ]
        return ToolResult(
            name=self.spec.name,
            ok=True,
            content="\n\n".join(lines),
            data={
                "hits": data_hits,
                "mode": context.mode,
                "evidence_status": context.evidence_status,
                "evidence_reason": context.evidence_reason,
                "rejected_hit_count": context.rejected_hit_count,
                "top_relevance": context.top_relevance,
                "relevance_margin": context.relevance_margin,
                "score_separation": context.score_separation,
                "query_coverage": context.query_coverage,
                "sparse_top_score": context.sparse_top_score,
                "dense_top_score": context.dense_top_score,
                "retrieval_state": context.retrieval_state,
                "retrieval_reason": context.retrieval_reason,
            },
        )


__all__ = ["MemorySearchTool"]
