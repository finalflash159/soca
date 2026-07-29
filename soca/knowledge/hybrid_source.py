from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Literal

from soca.knowledge.base import KnowledgeDocument, KnowledgeHit
from soca.knowledge.index.dense_persistence import DenseIndexStore
from soca.knowledge.index.models import VaultIndex
from soca.knowledge.index.vault_index import VaultIndexer, VaultIndexStore
from soca.knowledge.indexing.coordinator import IndexCoordinator
from soca.knowledge.indexing.identity import CorpusSpec
from soca.knowledge.indexing.status import DenseState, IndexStatus
from soca.knowledge.markdown_vault import (
    DEFAULT_EXCLUDE_DIRS,
    DEFAULT_EXCLUDE_FILES,
    DEFAULT_INCLUDE_GLOBS,
    MarkdownVaultKnowledgeSource,
    SearchScoringConfig,
)
from soca.knowledge.retriever import RankedHit
from soca.knowledge.retrievers.dense import DenseIndex, DenseRetriever, EmbeddingModel
from soca.knowledge.retrievers.rrf import reciprocal_rank_fusion
from soca.knowledge.retrievers.sparse_chunk import SparseChunkRetriever

LOGGER = logging.getLogger(__name__)
RetrievalMode = Literal["cached_sparse", "hybrid"]


class DenseUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class HybridConfig:
    rrf_k: int = 60
    per_retriever_limit: int = 12
    sparse_enabled: bool = True
    dense_enabled: bool = True
    dense_failure_policy: Literal["degrade", "raise"] = "degrade"

    def __post_init__(self) -> None:
        if (
            isinstance(self.rrf_k, bool)
            or not isinstance(self.rrf_k, int)
            or self.rrf_k < 1
            or isinstance(self.per_retriever_limit, bool)
            or not isinstance(self.per_retriever_limit, int)
            or self.per_retriever_limit < 1
        ):
            raise ValueError("hybrid retrieval limits must be positive")
        if not self.sparse_enabled and not self.dense_enabled:
            raise ValueError("at least one retriever must be enabled")
        if self.dense_failure_policy not in {"degrade", "raise"}:
            raise ValueError("unknown dense failure policy")


@dataclass(frozen=True)
class RetrievalBatch:
    hits: tuple[KnowledgeHit, ...]
    max_dense_score: float | None


class HybridKnowledgeSource(MarkdownVaultKnowledgeSource):
    def __init__(
        self,
        root: str | Path,
        *,
        model: EmbeddingModel | None,
        index_home: Path | None = None,
        config: HybridConfig | None = None,
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
        vault_store = VaultIndexStore(index_home=index_home)
        self._vault_indexer = VaultIndexer(self, vault_store)
        self._dense_store = DenseIndexStore(vault_store.manifest_path_for(self.root).parent)
        self._model = model
        self._config = config or HybridConfig()
        self._vault_index: VaultIndex | None = None
        self._dense_index: DenseIndex | None = None
        self._sparse_index: VaultIndex | None = None
        self._sparse: SparseChunkRetriever | None = None
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
                index_home=vault_store.index_home,
                model=model,
            )
            if lifecycle == "v2"
            else None
        )

    def _refresh_indexes(
        self,
    ) -> tuple[VaultIndex, DenseIndex | None, SparseChunkRetriever | None]:
        with self._index_lock:
            vault_index = self._vault_indexer.refresh(previous=self._vault_index)
            dense_index = self._dense_index
            self._ensure_dense_model_available()
            if self._config.dense_enabled and self._model is not None and vault_index.chunks:
                try:
                    dense_index = self._dense_store.refresh(
                        vault_index.chunks,
                        source_digest=vault_index.content_digest,
                        model=self._model,
                        previous=self._dense_index,
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    if self._dense_must_raise:
                        raise DenseUnavailableError("dense-only index refresh failed") from exc
                    LOGGER.warning(
                        "Dense index refresh failed; using sparse retrieval",
                        exc_info=True,
                    )
                    dense_index = None
            else:
                dense_index = None

            sparse = None
            if self._config.sparse_enabled:
                if self._sparse_index is not vault_index:
                    self._sparse = SparseChunkRetriever(vault_index.chunks, self.scoring)
                    self._sparse_index = vault_index
                sparse = self._sparse
            self._vault_index = vault_index
            self._dense_index = dense_index
            return vault_index, dense_index, sparse

    @property
    def _dense_must_raise(self) -> bool:
        return not self._config.sparse_enabled or self._config.dense_failure_policy == "raise"

    def _ensure_dense_model_available(self) -> None:
        if self._config.dense_enabled and self._model is None and self._dense_must_raise:
            raise DenseUnavailableError("dense-only retrieval has no model")

    def retrieve(self, query: str, *, limit: int = 5) -> RetrievalBatch:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be positive")
        if not query.strip():
            return RetrievalBatch((), None)

        if self._lifecycle == "v2":
            return self._retrieve_v2(query, limit=limit)

        vault_index, dense_index, sparse = self._refresh_indexes()
        rank_lists: list[list[RankedHit]] = []
        retriever_limit = max(limit, self._config.per_retriever_limit)
        if sparse is not None:
            rank_lists.append(sparse.rank(query, limit=retriever_limit))

        max_dense_score: float | None = None
        if self._config.dense_enabled and dense_index is not None and self._model is not None:
            try:
                dense_ranking = DenseRetriever(
                    dense_index,
                    self._model,
                ).rank_with_score(query, limit=retriever_limit)
            except (OSError, RuntimeError, ValueError) as exc:
                if self._dense_must_raise:
                    raise DenseUnavailableError("dense-only query failed") from exc
                LOGGER.warning("Dense query failed; using sparse retrieval", exc_info=True)
            else:
                rank_lists.append(list(dense_ranking.hits))
                max_dense_score = dense_ranking.max_score

        return self._build_batch(
            vault_index,
            reciprocal_rank_fusion(rank_lists, k=self._config.rrf_k)[:limit],
            max_dense_score,
        )

    def _retrieve_v2(self, query: str, *, limit: int) -> RetrievalBatch:
        assert self._coordinator is not None
        snapshot = self._coordinator.snapshot()
        vault_index = snapshot.sparse_index
        rank_lists: list[list[RankedHit]] = []
        retriever_limit = max(limit, self._config.per_retriever_limit)
        if self._config.sparse_enabled:
            rank_lists.append(
                SparseChunkRetriever(vault_index.chunks, self.scoring).rank(
                    query,
                    limit=retriever_limit,
                )
            )
        max_dense_score: float | None = None
        if (
            self._config.dense_enabled
            and snapshot.dense_state == DenseState.READY
            and snapshot.dense_index is not None
            and self._model is not None
        ):
            try:
                dense_ranking = DenseRetriever(snapshot.dense_index, self._model).rank_with_score(
                    query,
                    limit=retriever_limit,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                if self._dense_must_raise:
                    raise DenseUnavailableError("dense-only query failed") from exc
                LOGGER.warning("Dense query failed; using sparse retrieval", exc_info=True)
            else:
                rank_lists.append(list(dense_ranking.hits))
                max_dense_score = dense_ranking.max_score
        return self._build_batch(
            vault_index,
            reciprocal_rank_fusion(rank_lists, k=self._config.rrf_k)[:limit],
            max_dense_score,
        )

    @property
    def index_status(self) -> IndexStatus | None:
        return self._coordinator.status() if self._coordinator is not None else None

    def build_index(self, *, dense: bool = True) -> object:
        if self._coordinator is None:
            raise RuntimeError("index lifecycle v2 is required for explicit builds")
        return self._coordinator.build_blocking(dense=dense)

    def _build_batch(
        self,
        vault_index: VaultIndex,
        fused: tuple[tuple[str, float], ...],
        max_dense_score: float | None,
    ) -> RetrievalBatch:
        hits: list[KnowledgeHit] = []
        for chunk_id, score in fused:
            chunk = vault_index.chunk_by_id(chunk_id)
            if chunk is None:
                LOGGER.warning("Retriever returned an unknown chunk id")
                continue
            hits.append(
                KnowledgeHit(
                    document=KnowledgeDocument(
                        id=chunk.chunk_id,
                        path=chunk.document_path,
                        title=chunk.title,
                        text=chunk.text,
                        tags=chunk.tags,
                    ),
                    score=score,
                    snippet=chunk.text,
                    line_start=chunk.line_start,
                    line_end=chunk.line_end,
                )
            )
        return RetrievalBatch(tuple(hits), max_dense_score)

    def search(self, query: str, limit: int = 5) -> list[KnowledgeHit]:
        return list(self.retrieve(query, limit=limit).hits)
