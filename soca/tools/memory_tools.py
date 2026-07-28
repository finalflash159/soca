from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

from soca.memory.context import MemoryContextBuilder
from soca.memory.proposals import MemoryProposal, ProposalKind, ProposalStore
from soca.tools.base import SideEffectLevel, ToolResult, ToolSpec, object_schema


def _hit_payload(hit: object) -> dict[str, Any]:
    document = getattr(hit, "document", None)
    payload: dict[str, Any] = {
        "path": str(getattr(document, "path", "")),
        "title": str(getattr(document, "title", "")),
        "snippet": str(getattr(hit, "snippet", "")),
    }
    score = getattr(hit, "score", None)
    if score is not None:
        payload["score"] = score
    for field in ("line_start", "line_end"):
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
        )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        query = str(arguments["query"]).strip()
        if not query:
            raise ValueError("query must not be empty")
        limit = max(1, min(int(arguments.get("limit") or self.max_limit), self.max_limit))
        context = self.builder.build(query, include_archive=True)
        hits = tuple(context.hits)[:limit]
        data_hits = [_hit_payload(hit) for hit in hits]
        if not hits:
            return ToolResult(
                name=self.spec.name,
                ok=True,
                content="Mình chưa tìm thấy ghi chú phù hợp trong memory.",
                data={"hits": [], "mode": context.mode},
            )
        lines = [
            f"[M{index}] {item['title']} ({item['path']})\n{item['snippet']}"
            for index, item in enumerate(data_hits, start=1)
        ]
        return ToolResult(
            name=self.spec.name,
            ok=True,
            content="\n\n".join(lines),
            data={"hits": data_hits, "mode": context.mode},
        )


class MemoryProposeNoteTool:
    """Create a pending, immutable proposal; approval writes the markdown note."""

    def __init__(self, store: ProposalStore) -> None:
        self.store = store

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="memory.propose_note",
            description=(
                "Create a pending memory note proposal. It never writes an approved note; "
                "the user must approve it separately."
            ),
            input_schema=object_schema(
                properties={
                    "kind": {
                        "type": "string",
                        "enum": ["preference", "stable_fact", "project", "correction"],
                    },
                    "statement": {"type": "string"},
                    "evidence_excerpt": {"type": "string"},
                    "confidence": {"type": "number"},
                    "source_episode_id": {"type": "string"},
                },
                required=["kind", "statement", "evidence_excerpt"],
            ),
            side_effect=SideEffectLevel.LOCAL_STATE,
        )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        proposal_id = str(uuid4())
        source_episode_id = str(arguments.get("source_episode_id") or uuid4())
        kind_value = str(arguments["kind"])
        if kind_value not in {"preference", "stable_fact", "project", "correction"}:
            raise ValueError("unknown proposal kind")
        kind = cast(ProposalKind, kind_value)
        proposal = MemoryProposal(
            id=proposal_id,
            kind=kind,
            statement=str(arguments["statement"]).strip(),
            evidence_excerpt=str(arguments["evidence_excerpt"]).strip(),
            confidence=float(arguments.get("confidence", 0.8)),
            source_episode_id=source_episode_id,
            created_at=datetime.now(UTC),
        )
        self.store.put(proposal)
        return ToolResult(
            name=self.spec.name,
            ok=True,
            content=(
                f"Đã tạo proposal memory đang chờ duyệt: {proposal_id}. "
                "Chưa ghi note chính thức; cần approve proposal này trước."
            ),
            data={
                "proposal_id": proposal_id,
                "status": "pending",
                "source_episode_id": source_episode_id,
            },
        )


__all__ = ["MemorySearchTool", "MemoryProposeNoteTool"]
