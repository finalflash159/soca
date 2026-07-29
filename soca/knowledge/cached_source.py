from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Literal

from soca.knowledge.base import KnowledgeHit
from soca.knowledge.index.models import VaultIndex
from soca.knowledge.index.vault_index import VaultIndexer, VaultIndexStore
from soca.knowledge.indexing.coordinator import IndexCoordinator
from soca.knowledge.indexing.identity import CorpusSpec
from soca.knowledge.indexing.status import IndexStatus
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
        lifecycle: Literal["legacy", "v2"] = "legacy",
        corpus_kind: Literal["knowledge", "memory"] = "knowledge",
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
        if lifecycle not in {"legacy", "v2"}:
            raise ValueError("unknown index lifecycle")
        self._lifecycle = lifecycle
        self._coordinator = (
            IndexCoordinator(
                self,
                spec=CorpusSpec(
                    vault_path=self.root,
                    kind=corpus_kind,
                    include_globs=include_globs,
                    exclude_dirs=exclude_dirs,
                    exclude_files=exclude_files,
                    max_file_bytes=max_file_bytes,
                ),
                index_home=store.index_home,
                model=None,
            )
            if lifecycle == "v2"
            else None
        )

    @property
    def retrieval_mode(self) -> str:
        return "cached_sparse"

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
        if self._lifecycle == "v2":
            assert self._coordinator is not None
            snapshot = self._coordinator.snapshot()
            sparse = SparseDocumentRetriever(snapshot.sparse_index.documents, self.scoring)
            return sparse.search(query, limit=limit)
        _, sparse = self._refresh_index()
        return sparse.search(query, limit=limit)

    @property
    def index_status(self) -> IndexStatus | None:
        return self._coordinator.status() if self._coordinator is not None else None
