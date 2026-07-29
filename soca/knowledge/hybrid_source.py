from __future__ import annotations

import logging
from dataclasses import dataclass, field
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
from soca.knowledge.indexing.watcher import IndexWatcher
from soca.knowledge.markdown_vault import (
    DEFAULT_EXCLUDE_DIRS,
    DEFAULT_EXCLUDE_FILES,
    DEFAULT_INCLUDE_GLOBS,
    MarkdownVaultKnowledgeSource,
    SearchScoringConfig,
)
from soca.knowledge.retriever import RankedHit
from soca.knowledge.retrievers.bm25 import Bm25ChunkRetriever
from soca.knowledge.retrievers.dense import DenseIndex, DenseRetriever, EmbeddingModel
from soca.knowledge.retrievers.linear import linear_score_fusion
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
    sparse_backend: Literal["bm25", "lexical_custom"] = "bm25"
    fusion: Literal["linear", "rrf"] = "linear"
    dense_weight: float = 0.75

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
        if self.sparse_backend not in {"bm25", "lexical_custom"}:
            raise ValueError("unknown sparse backend")
        if self.fusion not in {"linear", "rrf"}:
            raise ValueError("unknown fusion mode")
        if (
            isinstance(self.dense_weight, bool)
            or not isinstance(self.dense_weight, (int, float))
            or not 0.0 <= float(self.dense_weight) <= 1.0
        ):
            raise ValueError("dense_weight must be in [0, 1]")


@dataclass(frozen=True)
class RetrievalDiagnostics:
    """Backend-local retrieval signals for the evidence contract."""

    sparse_state: str = "absent"
    dense_state: str = "absent"
    index_state: str = "unknown"
    sparse_top_score: float | None = None
    dense_top_score: float | None = None
    sparse_separation: float | None = None
    dense_separation: float | None = None
    query_coverage: float | None = None
    unavailable_reason: str = ""

    @property
    def overall_state(self) -> str:
        failed_states = {
            "model_missing",
            "missing",
            "stale",
            "incompatible",
            "failed",
            "unavailable",
            "queued",
            "building",
        }
        if (
            self.index_state == "empty"
            and not self.unavailable_reason
            and self.sparse_state not in failed_states
            and self.dense_state not in failed_states
        ):
            return "ready"
        sparse_usable = self.sparse_state == "ready"
        dense_usable = self.dense_state == "ready"
        if sparse_usable or dense_usable:
            if self.unavailable_reason:
                return "degraded"
            if self.sparse_state in {"stale", "failed", "unavailable"}:
                return "degraded"
            if self.dense_state in failed_states:
                return "degraded"
            return "ready"
        return "unavailable"


@dataclass(frozen=True)
class RetrievalBatch:
    hits: tuple[KnowledgeHit, ...]
    max_dense_score: float | None
    diagnostics: RetrievalDiagnostics = field(default_factory=RetrievalDiagnostics)


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
        self._last_dense_failure_reason = ""
        self._sparse_index: VaultIndex | None = None
        self._sparse: SparseChunkRetriever | Bm25ChunkRetriever | None = None
        self._index_lock = RLock()
        self._watcher: IndexWatcher | None = None
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
    ) -> tuple[
        VaultIndex,
        DenseIndex | None,
        SparseChunkRetriever | Bm25ChunkRetriever | None,
    ]:
        with self._index_lock:
            self._last_dense_failure_reason = ""
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
                    raise DenseUnavailableError("dense index refresh failed") from exc
            else:
                dense_index = None

            sparse = None
            if self._config.sparse_enabled:
                if self._sparse_index is not vault_index:
                    self._sparse = self._new_sparse_retriever(vault_index)
                    self._sparse_index = vault_index
                sparse = self._sparse
            self._vault_index = vault_index
            self._dense_index = dense_index
            return vault_index, dense_index, sparse

    @property
    def retrieval_mode(self) -> str:
        return "hybrid" if self._config.dense_enabled else "chunk_sparse"

    def _ensure_dense_model_available(self) -> None:
        if self._config.dense_enabled and self._model is None:
            raise DenseUnavailableError("hybrid retrieval has no model")

    def retrieve(self, query: str, *, limit: int = 5) -> RetrievalBatch:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be positive")
        if not query.strip():
            return RetrievalBatch((), None)

        if self._lifecycle == "v2":
            return self._retrieve_v2(query, limit=limit)

        vault_index, dense_index, sparse = self._refresh_indexes()
        sparse_hits: list[RankedHit] = []
        dense_hits: list[RankedHit] = []
        sparse_scores: dict[str, float] = {}
        dense_scores: dict[str, float] = {}
        sparse_state = "ready" if sparse is not None else "absent"
        dense_state = self._legacy_dense_state(dense_index, vault_index)
        unavailable_reason = self._last_dense_failure_reason
        if unavailable_reason:
            dense_state = "unavailable"
        retriever_limit = max(limit, self._config.per_retriever_limit)
        if sparse is not None:
            sparse_hits = sparse.rank(query, limit=retriever_limit)
            sparse_scores = {hit.chunk_id: hit.score for hit in sparse_hits}

        max_dense_score: float | None = None
        if self._config.dense_enabled and dense_index is not None and self._model is not None:
            try:
                dense_ranking = DenseRetriever(
                    dense_index,
                    self._model,
                ).rank_with_score(query, limit=retriever_limit)
            except (OSError, RuntimeError, ValueError) as exc:
                raise DenseUnavailableError("dense query failed") from exc
            else:
                dense_hits = list(dense_ranking.hits)
                dense_scores = {hit.chunk_id: hit.score for hit in dense_hits}
                max_dense_score = dense_ranking.max_score

        return self._build_batch(
            vault_index,
            self._fuse(sparse_hits, dense_hits)[:limit],
            max_dense_score,
            sparse_scores=sparse_scores,
            dense_scores=dense_scores,
            sparse_state=sparse_state,
            dense_state=dense_state,
            unavailable_reason=unavailable_reason,
        )

    def _retrieve_v2(self, query: str, *, limit: int) -> RetrievalBatch:
        assert self._coordinator is not None
        snapshot = self._coordinator.snapshot()
        vault_index = snapshot.sparse_index
        if not vault_index.chunks:
            return RetrievalBatch(
                (),
                None,
                RetrievalDiagnostics(
                    sparse_state="ready" if self._config.sparse_enabled else "absent",
                    dense_state="absent",
                    index_state="empty",
                ),
            )
        sparse_hits: list[RankedHit] = []
        dense_hits: list[RankedHit] = []
        sparse_scores: dict[str, float] = {}
        dense_scores: dict[str, float] = {}
        retriever_limit = max(limit, self._config.per_retriever_limit)
        if self._config.sparse_enabled:
            if self._sparse_index is not vault_index:
                self._sparse = self._new_sparse_retriever(vault_index)
                self._sparse_index = vault_index
            assert self._sparse is not None
            sparse_hits = self._sparse.rank(query, limit=retriever_limit)
            sparse_scores = {hit.chunk_id: hit.score for hit in sparse_hits}
        max_dense_score: float | None = None
        sparse_state = _enum_value(snapshot.sparse_state)
        dense_state = _enum_value(snapshot.dense_state)
        unavailable_reason = ""
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
                raise DenseUnavailableError("dense query failed") from exc
            else:
                dense_hits = list(dense_ranking.hits)
                dense_scores = {hit.chunk_id: hit.score for hit in dense_hits}
                max_dense_score = dense_ranking.max_score
        elif self._config.dense_enabled:
            raise DenseUnavailableError(f"dense index is not ready: {dense_state}")
        return self._build_batch(
            vault_index,
            self._fuse(sparse_hits, dense_hits)[:limit],
            max_dense_score,
            sparse_scores=sparse_scores,
            dense_scores=dense_scores,
            sparse_state=sparse_state if self._config.sparse_enabled else "absent",
            dense_state=dense_state if self._config.dense_enabled else "absent",
            unavailable_reason=unavailable_reason,
        )

    @property
    def index_status(self) -> IndexStatus | None:
        return self._coordinator.status() if self._coordinator is not None else None

    def build_index(self, *, dense: bool = True) -> object:
        if self._coordinator is None:
            raise RuntimeError("index lifecycle v2 is required for explicit builds")
        return self._coordinator.build_blocking(dense=dense)

    def activate_watcher(self, *, interval_seconds: float = 2.0) -> None:
        if self._coordinator is None:
            raise RuntimeError("index lifecycle v2 is required for watcher")
        if self._model is None:
            raise DenseUnavailableError("hybrid retrieval has no model")
        if self._watcher is not None:
            return
        watcher = IndexWatcher(self._coordinator, interval_seconds=interval_seconds)
        watcher.reconcile()
        watcher.start()
        self._watcher = watcher

    def close(self) -> None:
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None

    def _build_batch(
        self,
        vault_index: VaultIndex,
        fused: tuple[tuple[str, float], ...],
        max_dense_score: float | None,
        *,
        sparse_scores: dict[str, float],
        dense_scores: dict[str, float],
        sparse_state: str,
        dense_state: str,
        unavailable_reason: str,
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
                    retrieval_backend=(
                        "hybrid"
                        if chunk_id in sparse_scores and chunk_id in dense_scores
                        else "dense"
                        if chunk_id in dense_scores
                        else self._config.sparse_backend
                    ),
                    sparse_score=sparse_scores.get(chunk_id),
                    dense_score=dense_scores.get(chunk_id),
                    fusion_score=score,
                )
            )
        return RetrievalBatch(
            tuple(hits),
            max_dense_score,
            RetrievalDiagnostics(
                sparse_state=sparse_state,
                dense_state=dense_state,
                index_state="ready" if vault_index.chunks else "empty",
                sparse_top_score=_top_score(sparse_scores),
                dense_top_score=_top_score(dense_scores),
                sparse_separation=_score_separation(sparse_scores),
                dense_separation=_score_separation(dense_scores),
                unavailable_reason=unavailable_reason,
            ),
        )

    def _legacy_dense_state(
        self,
        dense_index: DenseIndex | None,
        vault_index: VaultIndex,
    ) -> str:
        if not self._config.dense_enabled:
            return "absent"
        if self._model is None:
            return "missing"
        if not vault_index.chunks:
            return "absent"
        return "ready" if dense_index is not None else "unavailable"

    def search(self, query: str, limit: int = 5) -> list[KnowledgeHit]:
        return list(self.retrieve(query, limit=limit).hits)

    def _new_sparse_retriever(
        self,
        vault_index: VaultIndex,
    ) -> SparseChunkRetriever | Bm25ChunkRetriever:
        if self._config.sparse_backend == "bm25":
            return Bm25ChunkRetriever(vault_index.chunks)
        return SparseChunkRetriever(vault_index.chunks, self.scoring)

    def _fuse(
        self,
        sparse_hits: list[RankedHit],
        dense_hits: list[RankedHit],
    ) -> tuple[tuple[str, float], ...]:
        if self._config.fusion == "linear":
            return linear_score_fusion(
                sparse_hits,
                dense_hits,
                dense_weight=self._config.dense_weight,
            )
        lists = tuple(hits for hits in (sparse_hits, dense_hits) if hits)
        return reciprocal_rank_fusion(lists, k=self._config.rrf_k)


def _enum_value(value: object) -> str:
    candidate = getattr(value, "value", value)
    return str(candidate)


def _top_score(scores: dict[str, float]) -> float | None:
    return next(iter(scores.values()), None)


def _score_separation(scores: dict[str, float]) -> float | None:
    values = tuple(scores.values())
    if len(values) < 2:
        return None
    return float(values[0] - values[1])
