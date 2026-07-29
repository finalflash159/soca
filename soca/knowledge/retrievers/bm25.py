from __future__ import annotations

import numpy as np

from soca.knowledge.index.models import MarkdownChunk
from soca.knowledge.markdown_vault import tokenize_terms
from soca.knowledge.retriever import RankedHit


class Bm25ChunkRetriever:
    backend = "bm25"

    def __init__(self, chunks: tuple[MarkdownChunk, ...]) -> None:
        import bm25s

        self.chunks = chunks
        self._chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
        self._retriever = bm25s.BM25(method="lucene")
        self._retriever.index(
            [list(tokenize_terms(chunk.text)) for chunk in chunks],
            show_progress=False,
        )

    @property
    def available(self) -> bool:
        return bool(self._chunk_ids)

    def rank(self, query: str, *, limit: int) -> list[RankedHit]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be positive")
        terms = list(tokenize_terms(query))
        if not terms or not self._chunk_ids:
            return []
        result = self._retriever.retrieve(
            [terms],
            k=min(limit, len(self._chunk_ids)),
            show_progress=False,
        )
        indices = np.asarray(result[0][0], dtype=np.int64)
        scores = np.asarray(result[1][0], dtype=np.float32)
        return [
            RankedHit(
                chunk_id=self._chunk_ids[int(index)],
                rank=rank,
                score=float(score),
            )
            for rank, (index, score) in enumerate(
                zip(indices, scores, strict=True),
                start=1,
            )
        ]
