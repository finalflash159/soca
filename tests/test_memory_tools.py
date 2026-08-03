from __future__ import annotations

from soca.knowledge import KnowledgeDocument, KnowledgeHit
from soca.memory import MemoryContextBuilder, MemoryRetrievalResult
from soca.tools import MemorySearchTool


class _MemorySource:
    def read_core(self) -> str:
        return ""

    def retrieve_archive(self, query: str) -> MemoryRetrievalResult:
        hit = KnowledgeHit(
            KnowledgeDocument("memory/decision.md", "memory/decision.md", "Decision", "TTS local"),
            score=0.9,
            snippet="TTS local vì riêng tư.",
        )
        return MemoryRetrievalResult(text=hit.snippet, hits=(hit,), mode="retrieved")


def test_memory_search_returns_memory_hits_only() -> None:
    builder = MemoryContextBuilder(long_term=_MemorySource())
    result = MemorySearchTool(builder).run({"query": "TTS"})

    assert result.ok is True
    assert result.data["hits"][0]["path"] == "memory/decision.md"
    assert "TTS local" in result.content
    assert result.data["evidence_status"] == "weak"


class _ScoredMemorySource:
    def read_core(self) -> str:
        return ""

    def retrieve_archive(self, query: str) -> MemoryRetrievalResult:
        relevant = KnowledgeHit(
            KnowledgeDocument(
                "memory/tts.md",
                "memory/tts.md",
                "TTS decision",
                "TTS local vì riêng tư.",
            ),
            score=10.0,
            snippet="TTS local vì riêng tư.",
            retrieval_backend="lexical_custom",
            sparse_score=100.0,
        )
        distractor = KnowledgeHit(
            KnowledgeDocument(
                "memory/weather.md",
                "memory/weather.md",
                "Weather",
                "Thời tiết không liên quan.",
            ),
            score=1.0,
            snippet="Thời tiết không liên quan.",
            retrieval_backend="lexical_custom",
            sparse_score=10.0,
        )
        return MemoryRetrievalResult(
            text="raw memory must not bypass the gate",
            hits=(relevant, distractor),
            mode="retrieved",
        )


def test_memory_context_admits_known_hits_and_rejects_distractors() -> None:
    context = MemoryContextBuilder(long_term=_ScoredMemorySource()).build("TTS")

    assert context.evidence_status == "supported"
    assert context.rejected_hit_count == 1
    assert [hit.document.path for hit in context.hits] == ["memory/tts.md"]
    assert "memory/weather.md" not in context.prompt_text
