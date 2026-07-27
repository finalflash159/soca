from __future__ import annotations

from pathlib import Path

from soca.knowledge import KnowledgeDocument, KnowledgeHit
from soca.memory import (
    MarkdownLongTermMemory,
    MemoryContextBuilder,
    RetrievedMemory,
    RetrievedMemoryConfig,
)


class FakeSource:
    def __init__(self, hits: list[KnowledgeHit]) -> None:
        self.hits = hits

    def search(self, query: str, limit: int = 5) -> list[KnowledgeHit]:
        del query
        return self.hits[:limit]

    def read(self, path: str) -> KnowledgeDocument:
        return self.hits[0].document if path == self.hits[0].document.path else self.hits[1].document


def test_retrieved_memory_returns_ranked_marked_context(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "profile.md").write_text("fallback", encoding="utf-8")
    docs = [
        KnowledgeDocument("a", "memory/a.md", "A", "# A\nTTS choice"),
        KnowledgeDocument("b", "memory/b.md", "B", "# B\nDistractor"),
    ]
    source = FakeSource([KnowledgeHit(docs[0], 2.0, "TTS choice", 1, 2), KnowledgeHit(docs[1], 1.0, "Distractor", 1, 2)])
    memory = RetrievedMemory(
        source,
        MarkdownLongTermMemory(root),
        config=RetrievedMemoryConfig(top_k=1),
    )
    context = MemoryContextBuilder(long_term=memory).build("TTS")
    assert context.mode == "retrieved"
    assert "[M1] memory/a.md:1-2" in context.prompt_text
    assert "Distractor" not in context.prompt_text
