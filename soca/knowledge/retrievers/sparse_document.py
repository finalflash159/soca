from __future__ import annotations

from soca.knowledge.base import KnowledgeDocument, KnowledgeHit
from soca.knowledge.markdown_vault import (
    LexicalSnapshot,
    MarkdownVaultKnowledgeSource,
    SearchScoringConfig,
    prepare_lexical_snapshot,
)
from soca.knowledge.retriever import RankedHit


class SparseDocumentRetriever:
    def __init__(
        self,
        documents: tuple[KnowledgeDocument, ...],
        scoring: SearchScoringConfig,
    ) -> None:
        self.documents = documents
        self.scoring = scoring
        self.snapshot = prepare_lexical_snapshot(documents)

    @property
    def available(self) -> bool:
        return bool(self.documents)

    def rank(self, query: str, *, limit: int) -> list[RankedHit]:
        source = _ScoringSource(self.scoring)
        hits = source.search_snapshot(query, self.snapshot, limit=limit)
        return [
            RankedHit(
                chunk_id=hit.document.id,
                rank=rank,
                score=hit.score,
            )
            for rank, hit in enumerate(hits, start=1)
        ]

    def search(self, query: str, *, limit: int) -> list[KnowledgeHit]:
        return _ScoringSource(self.scoring).search_snapshot(
            query,
            self.snapshot,
            limit=limit,
        )


class _ScoringSource(MarkdownVaultKnowledgeSource):
    def __init__(self, scoring: SearchScoringConfig) -> None:
        self.scoring = scoring

    def search_snapshot(
        self,
        query: str,
        snapshot: LexicalSnapshot,
        *,
        limit: int,
    ) -> list[KnowledgeHit]:
        return self._search_lexical_snapshot(query, snapshot, limit=limit)
