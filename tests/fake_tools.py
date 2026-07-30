from __future__ import annotations

from typing import Any

from soca.tools import ToolResult, ToolSpec, object_schema


class ReadOnlyInspectTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="knowledge.inspect",
            description="Return navigation metadata only.",
            input_schema=object_schema(),
            workflow_capability="knowledge_catalog",
        )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        del arguments
        return ToolResult(
            name=self.spec.name,
            ok=True,
            content='{"documents":[{"path":"wiki/index.md","title":"Index"}]}',
            data={
                "documents": [{"path": "wiki/index.md", "title": "Index"}],
                "relations": [],
                "metadata_only": True,
                "evidence_status": "not_requested",
            },
        )


class ReadOnlySearchTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="knowledge.search",
            description="Search test knowledge evidence.",
            input_schema=object_schema(
                properties={
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                required=["query"],
            ),
            workflow_capability="knowledge_retrieval",
        )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        del arguments
        return ToolResult(
            name=self.spec.name,
            ok=True,
            content="no test evidence",
            data={"hits": [], "evidence_status": "insufficient"},
        )
