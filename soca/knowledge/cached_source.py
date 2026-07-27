from __future__ import annotations

from pathlib import Path
from threading import RLock

from soca.knowledge.base import KnowledgeHit
from soca.knowledge.index.models import VaultIndex
from soca.knowledge.index.vault_index import VaultIndexer, VaultIndexStore
from soca.knowledge.markdown_vault import (
    DEFAULT_EXCLUDE_DIRS,
    DEFAULT_EXCLUDE_FILES,
    DEFAULT_INCLUDE_GLOBS,
    MarkdownVaultKnowledgeSource,
    SearchScoringConfig,
)
from soca.knowledge.retrievers.sparse_document import SparseDocumentRetriever


class CachedMarkdownVaultKnowledgeSource(MarkdownVaultKnowledgeSource):
    def __init__(
        self,
        root: str | Path,
        *,
        index_home: Path | None = None,
        exclude_dirs: tuple[str, ...] = DEFAULT_EXCLUDE_DIRS,
        exclude_files: tuple[str, ...] = DEFAULT_EXCLUDE_FILES,
        include_globs: tuple[str, ...] = DEFAULT_INCLUDE_GLOBS,
        max_file_bytes: int = 256 * 1024,
        scoring: SearchScoringConfig | None = None,
    ) -> None:
        super().__init__(
            root,
            exclude_dirs=exclude_dirs,
            exclude_files=exclude_files,
            include_globs=include_globs,
            max_file_bytes=max_file_bytes,
            scoring=scoring,
        )
        store = VaultIndexStore(index_home=index_home)
        self._indexer = VaultIndexer(self, store)
        self._index: VaultIndex | None = None
        self._sparse_index: VaultIndex | None = None
        self._sparse: SparseDocumentRetriever | None = None
        self._index_lock = RLock()

    def _refresh_index(
        self,
    ) -> tuple[VaultIndex, SparseDocumentRetriever]:
        with self._index_lock:
            index = self._indexer.refresh(previous=self._index)
            self._index = index
            if self._sparse_index is not index:
                self._sparse = SparseDocumentRetriever(
                    index.documents,
                    self.scoring,
                )
                self._sparse_index = index
            assert self._sparse is not None
            return index, self._sparse

    def search(self, query: str, limit: int = 5) -> list[KnowledgeHit]:
        _, sparse = self._refresh_index()
        return sparse.search(query, limit=limit)
