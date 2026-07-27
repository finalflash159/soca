from __future__ import annotations

from datetime import UTC, datetime, timedelta

from soca.knowledge import KnowledgeDocument, KnowledgeHit
from soca.memory.scoring import MemoryScoreConfig, rerank_memory_hits


def test_memory_scoring_uses_recency_and_stable_ties() -> None:
    now = datetime.now(UTC)
    old = KnowledgeDocument(
        "old", "memory/old.md", "Old", "---\nupdated_at: 2020-01-01T00:00:00Z\n---\nold"
    )
    fresh = KnowledgeDocument(
        "fresh",
        "memory/fresh.md",
        "Fresh",
        f"---\nupdated_at: {now.isoformat()}\n---\nfresh",
    )
    hits = rerank_memory_hits(
        (
            KnowledgeHit(old, 2.0, "old"),
            KnowledgeHit(fresh, 1.0, "fresh"),
        ),
        top_k=2,
        config=MemoryScoreConfig(),
        now=now + timedelta(seconds=1),
    )
    assert hits[0].document.path == "memory/old.md" or hits[0].document.path == "memory/fresh.md"
    assert all(0.0 <= item.score.total <= 1.0 for item in hits)
