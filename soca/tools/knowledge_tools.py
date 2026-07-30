from __future__ import annotations

from typing import Any

from soca.knowledge import KnowledgeSource
from soca.tools.base import (
    InvalidToolInput,
    SideEffectLevel,
    ToolExecutionStatus,
    ToolResult,
    ToolSpec,
    object_schema,
)


def _retrieval_metadata(diagnostics: Any | None) -> dict[str, Any]:
    if diagnostics is None:
        return {}
    metadata: dict[str, Any] = {}
    for field in (
        "sparse_state",
        "dense_state",
        "index_state",
        "sparse_top_score",
        "dense_top_score",
        "sparse_separation",
        "dense_separation",
        "query_coverage",
        "unavailable_reason",
    ):
        value = getattr(diagnostics, field, None)
        if value not in (None, ""):
            metadata[field] = value
    metadata["retrieval_state"] = str(getattr(diagnostics, "overall_state", "ready"))
    return metadata


class KnowledgeSearchTool:
    def __init__(
        self,
        source: KnowledgeSource,
        max_limit: int = 5,
    ) -> None:
        self.source = source
        self.max_limit = max_limit

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="knowledge.search",
            description="Search local wiki markdown notes and return matching snippets.",
            input_schema=object_schema(
                properties={
                    "query": {
                        "type": "string",
                        "description": "Search query.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of hits to return.",
                    },
                },
                required=["query"],
            ),
            side_effect=SideEffectLevel.READ_ONLY,
            workflow_capability="knowledge_search",
        )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        query = str(arguments["query"]).strip()
        if not query:
            raise InvalidToolInput("empty_query")

        limit = int(arguments.get("limit") or self.max_limit)
        limit = max(1, min(limit, self.max_limit))
        retrieve = getattr(self.source, "retrieve", None)
        diagnostics: Any | None = None
        if callable(retrieve):
            batch = retrieve(query, limit=limit)
            hits = list(getattr(batch, "hits", ()))
            diagnostics = getattr(batch, "diagnostics", None)
        else:
            hits = self.source.search(query, limit=limit)

        if not hits:
            data: dict[str, Any] = {"hits": []}
            data.update(_retrieval_metadata(diagnostics))
            return ToolResult(
                name=self.spec.name,
                ok=True,
                content="Mình chưa tìm thấy thông tin đó trong knowledge vault.",
                data=data,
            )

        lines: list[str] = []
        data_hits: list[dict[str, Any]] = []
        for index, hit in enumerate(hits, start=1):
            lines.append(f"[K{index}] {hit.document.title} ({hit.document.path})\n{hit.snippet}")
            data_hit: dict[str, Any] = {
                "path": hit.document.path,
                "title": hit.document.title,
                "score": hit.score,
                "snippet": hit.snippet,
                "retrieval_backend": hit.retrieval_backend,
            }
            for field in ("sparse_score", "dense_score", "fusion_score"):
                value = getattr(hit, field, None)
                if value is not None:
                    data_hit[field] = value
            if hit.line_start is not None:
                data_hit["line_start"] = hit.line_start
                data_hit["line_end"] = hit.line_end
            data_hits.append(data_hit)

        metadata: dict[str, Any] = {"hits": data_hits}
        metadata.update(_retrieval_metadata(diagnostics))
        return ToolResult(
            name=self.spec.name,
            ok=True,
            content="\n\n".join(lines),
            data=metadata,
        )


class KnowledgeReadTool:
    def __init__(
        self,
        source: KnowledgeSource,
        max_chars: int = 4000,
    ) -> None:
        self.source = source
        self.max_chars = max_chars

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="knowledge.read",
            description="Read one local wiki markdown note by relative path.",
            input_schema=object_schema(
                properties={
                    "path": {
                        "type": "string",
                        "description": "Vault-relative markdown path.",
                    }
                },
                required=["path"],
            ),
            side_effect=SideEffectLevel.READ_ONLY,
            workflow_capability="knowledge_read",
        )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        path = str(arguments["path"]).strip()
        if not path:
            raise InvalidToolInput("empty_path")

        try:
            doc = self.source.read(path)
        except FileNotFoundError:
            return ToolResult(
                name=self.spec.name,
                ok=False,
                content="",
                error="not_found",
                status=ToolExecutionStatus.NOT_FOUND,
            )
        except ValueError:
            return ToolResult(
                name=self.spec.name,
                ok=False,
                content="",
                error="invalid_path",
                status=ToolExecutionStatus.INVALID,
            )
        text = doc.text.strip()
        truncated = False
        if len(text) > self.max_chars:
            text = text[: max(0, self.max_chars - 3)].rstrip() + "..."
            truncated = True

        return ToolResult(
            name=self.spec.name,
            ok=True,
            content=f"# {doc.title}\n\n{text}",
            data={
                "path": doc.path,
                "title": doc.title,
                "tags": list(doc.tags),
                "truncated": truncated,
            },
        )
