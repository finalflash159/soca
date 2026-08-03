from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Literal

from soca.knowledge.base import KnowledgeHit
from soca.knowledge.catalog import CatalogIndexSnapshot
from soca.knowledge.index.vault_index import VaultIndexStore
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
        self._index_lock = RLock()
        self._coordinator = IndexCoordinator(
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

    @property
    def retrieval_mode(self) -> str:
        return "cached_sparse"

    def search(self, query: str, limit: int = 5) -> list[KnowledgeHit]:
        with self._index_lock:
            snapshot = self._coordinator.snapshot()
            sparse = SparseDocumentRetriever(snapshot.sparse_index.documents, self.scoring)
            return sparse.search(query, limit=limit)

    def catalog_index_snapshot(self) -> CatalogIndexSnapshot:
        with self._index_lock:
            snapshot = self._coordinator.snapshot()
            return CatalogIndexSnapshot(
                revision=snapshot.revision,
                index=snapshot.sparse_index,
            )

    @property
    def index_status(self) -> IndexStatus | None:
        return self._coordinator.status()

    def build_index(self, *, dense: bool = False) -> object:
        if dense:
            raise ValueError("cached_sparse does not have a dense index")
        return self._coordinator.build_blocking(dense=False)
