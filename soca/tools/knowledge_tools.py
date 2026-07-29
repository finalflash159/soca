from __future__ import annotations

from typing import Any

from soca.knowledge import KnowledgeSource
from soca.tools.base import SideEffectLevel, ToolResult, ToolSpec, object_schema


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
        )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        query = str(arguments["query"]).strip()
        if not query:
            raise ValueError("query must not be empty")

        limit = int(arguments.get("limit") or self.max_limit)
        limit = max(1, min(limit, self.max_limit))
        hits = self.source.search(query, limit=limit)

        if not hits:
            return ToolResult(
                name=self.spec.name,
                ok=True,
                content="Mình chưa tìm thấy thông tin đó trong knowledge vault.",
                data={"hits": []},
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

        return ToolResult(
            name=self.spec.name,
            ok=True,
            content="\n\n".join(lines),
            data={"hits": data_hits},
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
        )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        path = str(arguments["path"]).strip()
        if not path:
            raise ValueError("path must not be empty")

        doc = self.source.read(path)
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
