from __future__ import annotations

from typing import Any

from soca.tools import ToolResult, ToolSpec, object_schema


class ReadOnlyCatalogTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="knowledge.catalog",
            description="Return a deterministic test knowledge catalog.",
            input_schema=object_schema(),
            workflow_capability="knowledge_catalog",
        )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        del arguments
        return ToolResult(
            name=self.spec.name,
            ok=True,
            content=(
                '{"documents":[{"label":"K1","path":"wiki/index.md",'
                '"title":"Index","summary":"Knowledge catalog index."}]}'
            ),
            data={
                "hits": [
                    {
                        "path": "wiki/index.md",
                        "title": "Index",
                        "snippet": "Knowledge catalog index.",
                    }
                ]
            },
        )


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
                }
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
