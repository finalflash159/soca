from __future__ import annotations

from datetime import UTC
from pathlib import Path

from soca.knowledge import KnowledgeDocument, KnowledgeHit
from soca.memory import MemoryContextBuilder, MemoryProfileResult, ProposalStore
from soca.tools import MemoryProposeNoteTool, MemorySearchTool


class _MemorySource:
    def read_profile(self) -> str:
        return ""

    def retrieve_profile(self, query: str) -> MemoryProfileResult:
        hit = KnowledgeHit(
            KnowledgeDocument("memory/decision.md", "memory/decision.md", "Decision", "TTS local"),
            score=0.9,
            snippet="TTS local vì riêng tư.",
        )
        return MemoryProfileResult(text=hit.snippet, hits=(hit,), mode="retrieved")


def test_memory_search_returns_memory_hits_only() -> None:
    builder = MemoryContextBuilder(long_term=_MemorySource())
    result = MemorySearchTool(builder).run({"query": "TTS"})

    assert result.ok is True
    assert result.data["hits"][0]["path"] == "memory/decision.md"
    assert "TTS local" in result.content


def test_propose_note_creates_pending_proposal_without_approved_note(tmp_path: Path) -> None:
    store = ProposalStore(tmp_path / ".proposals")
    result = MemoryProposeNoteTool(store).run(
        {
            "kind": "preference",
            "statement": "Ưu tiên chạy local vì riêng tư.",
            "evidence_excerpt": "Tôi muốn dữ liệu ở trên máy.",
        }
    )

    assert result.ok is True
    assert result.data["status"] == "pending"
    pending = store.list(status="pending")
    assert len(pending) == 1
    assert pending[0].created_at.tzinfo is not None
    assert pending[0].created_at.astimezone(UTC) == pending[0].created_at
    assert not (tmp_path / "memory" / "captured").exists()
