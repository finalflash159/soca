from __future__ import annotations

from soca.knowledge.index.models import MarkdownChunk
from soca.knowledge.markdown_vault import SearchScoringConfig
from soca.knowledge.retriever import RankedHit
from soca.knowledge.retrievers.sparse_document import SparseDocumentRetriever


class SparseChunkRetriever:
    def __init__(
        self,
        chunks: tuple[MarkdownChunk, ...],
        scoring: SearchScoringConfig,
    ) -> None:
        self.chunks = chunks
        self._delegate = SparseDocumentRetriever(
            tuple(chunk.as_document() for chunk in chunks),
            scoring,
        )

    @property
    def available(self) -> bool:
        return self._delegate.available

    def rank(self, query: str, *, limit: int) -> list[RankedHit]:
        return self._delegate.rank(query, limit=limit)
