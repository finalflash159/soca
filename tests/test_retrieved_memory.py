from __future__ import annotations

from pathlib import Path

from soca.knowledge import KnowledgeDocument, KnowledgeHit
from soca.knowledge.relevance import RelevancePolicy
from soca.memory import (
    CoreMemoryStore,
    MemoryContextBuilder,
    RetrievedMemory,
    RetrievedMemoryConfig,
)


class FakeSource:
    def __init__(self, hits: list[KnowledgeHit]) -> None:
        self.hits = hits
        self.closed = False

    def search(self, query: str, limit: int = 5) -> list[KnowledgeHit]:
        del query
        return self.hits[:limit]

    def read(self, path: str) -> KnowledgeDocument:
        return self.hits[0].document if path == self.hits[0].document.path else self.hits[1].document

    def close(self) -> None:
        self.closed = True


def test_retrieved_memory_returns_ranked_marked_context(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "core.json").write_text(
        '{"schema_version":1,"items":[]}', encoding="utf-8"
    )
    docs = [
        KnowledgeDocument("a", "memory/a.md", "A", "# A\nTTS choice"),
        KnowledgeDocument("b", "memory/b.md", "B", "# B\nDistractor"),
    ]
    source = FakeSource([KnowledgeHit(docs[0], 2.0, "TTS choice", 1, 2), KnowledgeHit(docs[1], 1.0, "Distractor", 1, 2)])
    memory = RetrievedMemory(
        source,
        CoreMemoryStore(root),
        config=RetrievedMemoryConfig(top_k=1),
    )
    context = MemoryContextBuilder(long_term=memory).build("TTS")
    assert context.mode == "retrieved"
    assert "[M1] memory/a.md:1-2" in context.prompt_text
    assert "Distractor" not in context.prompt_text


def test_retrieved_memory_gates_before_memory_top_k_rerank(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "core.json").write_text(
        '{"schema_version":1,"items":[]}', encoding="utf-8"
    )
    low = KnowledgeDocument("low", "memory/low.md", "Low", "TTS low confidence")
    high = KnowledgeDocument("high", "memory/high.md", "High", "TTS selected")
    source = FakeSource(
        [
            KnowledgeHit(
                low,
                0.99,
                low.text,
                dense_score=0.70,
                retrieval_backend="dense",
            ),
            KnowledgeHit(
                high,
                0.80,
                high.text,
                dense_score=0.90,
                retrieval_backend="dense",
            ),
        ]
    )
    memory = RetrievedMemory(
        source,
        CoreMemoryStore(root),
        config=RetrievedMemoryConfig(top_k=1),
        relevance_policy=RelevancePolicy(min_dense_score=0.85),
    )

    context = MemoryContextBuilder(long_term=memory).build("what TTS was selected")

    assert "memory/high.md" in context.prompt_text
    assert "memory/low.md" not in context.prompt_text


def test_retrieved_memory_closes_owned_index_source(tmp_path: Path) -> None:
    root = tmp_path / "vault"
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "core.json").write_text(
        '{"schema_version":1,"items":[]}', encoding="utf-8"
    )
    document = KnowledgeDocument("memory", "memory/note.md", "Note", "TTS choice")
    source = FakeSource([KnowledgeHit(document, 1.0, document.text, 1, 1)])
    memory = RetrievedMemory(source, CoreMemoryStore(root))

    memory.close()

    assert source.closed is True
