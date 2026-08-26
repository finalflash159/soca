from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from soca.knowledge.base import KnowledgeSource
from soca.knowledge.cached_source import CachedMarkdownVaultKnowledgeSource
from soca.knowledge.hybrid_source import HybridConfig, HybridKnowledgeSource
from soca.knowledge.index.persistence import default_index_home
from soca.knowledge.indexing.models import load_model
from soca.knowledge.retrievers.dense import (
    AITEAMVN_V2_MODEL,
    DeferredEmbeddingModel,
    production_embedding_fingerprint,
)

RetrievalMode = Literal["cached_sparse", "chunk_sparse", "hybrid"]
DenseBackend = Literal["aiteamvn_v2"]
SparseBackend = Literal["bm25", "lexical_custom"]
FusionMode = Literal["linear", "rrf"]


@dataclass(frozen=True)
class RetrievalConfig:
    mode: RetrievalMode = "hybrid"
    dense_backend: str = "aiteamvn_v2"
    sparse_backend: SparseBackend = "bm25"
    fusion: FusionMode = "linear"
    dense_weight: float = 0.75
    rrf_k: int = 60
    per_retriever_limit: int = 12
    watcher_enabled: bool = True
    watcher_interval_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.mode not in {"cached_sparse", "chunk_sparse", "hybrid"}:
            raise ValueError("unknown retrieval mode")
        if self.mode == "hybrid" and self.dense_backend != "aiteamvn_v2":
            raise ValueError("unknown dense backend")
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
        if not isinstance(self.watcher_enabled, bool):
            raise ValueError("watcher_enabled must be a boolean")
        if (
            isinstance(self.watcher_interval_seconds, bool)
            or not isinstance(self.watcher_interval_seconds, (int, float))
            or self.watcher_interval_seconds <= 0
        ):
            raise ValueError("watcher interval must be positive")
        if (
            isinstance(self.rrf_k, bool)
            or not isinstance(self.rrf_k, int)
            or self.rrf_k < 1
            or isinstance(self.per_retriever_limit, bool)
            or not isinstance(self.per_retriever_limit, int)
            or self.per_retriever_limit < 1
        ):
            raise ValueError("retrieval limits must be positive")


def _build_model(backend: str):
    if backend != "aiteamvn_v2":
        raise ValueError("unknown production dense backend")
    return load_model("aiteamvn-v2")


def build_retrieval_source(
    vault: Path,
    *,
    include_globs: tuple[str, ...],
    config: RetrievalConfig | None = None,
    index_home: Path | None = None,
    defer_dense_model: bool = False,
) -> KnowledgeSource:
    resolved = config or RetrievalConfig()
    corpus_kind = "memory" if include_globs == ("memory/**/*.md",) else "knowledge"
    resolved_index_home = index_home or default_index_home(vault)
    common = {
        "index_home": resolved_index_home,
        "include_globs": include_globs,
        "corpus_kind": corpus_kind,
    }
    if resolved.mode == "cached_sparse":
        return CachedMarkdownVaultKnowledgeSource(vault, **common)
    if resolved.mode == "chunk_sparse":
        return HybridKnowledgeSource(
            vault,
            model=None,
            config=HybridConfig(
                rrf_k=resolved.rrf_k,
                per_retriever_limit=resolved.per_retriever_limit,
                sparse_enabled=True,
                dense_enabled=False,
                sparse_backend=resolved.sparse_backend,
                fusion=resolved.fusion,
                dense_weight=resolved.dense_weight,
            ),
            **common,
        )

    if defer_dense_model:
        if resolved.dense_backend != "aiteamvn_v2":
            raise ValueError("unknown production dense backend")
        model = DeferredEmbeddingModel(
            model_id=f"sentence_transformers:{AITEAMVN_V2_MODEL}",
            embedding_fingerprint=production_embedding_fingerprint(),
            loader=lambda: _build_model(resolved.dense_backend),
        )
    else:
        model = _build_model(resolved.dense_backend)

    source = HybridKnowledgeSource(
        vault,
        model=model,
        config=HybridConfig(
            rrf_k=resolved.rrf_k,
            per_retriever_limit=resolved.per_retriever_limit,
            sparse_backend=resolved.sparse_backend,
            fusion=resolved.fusion,
            dense_weight=resolved.dense_weight,
        ),
        **common,
    )
    # A deferred model cannot safely start a dense rebuild in the background:
    # that would immediately materialize the weights we intentionally deferred.
    # Snapshots still reconcile sparse content synchronously, and an explicit
    # retrieval/index operation loads the same pinned dense model on demand.
    if resolved.watcher_enabled and not defer_dense_model:
        try:
            source.activate_watcher(interval_seconds=resolved.watcher_interval_seconds)
        except Exception:
            try:
                source.close()
            except Exception as cleanup_exc:  # noqa: BLE001 - preserve cleanup failure
                raise RuntimeError("retrieval source startup cleanup failed") from cleanup_exc
            raise
    return source


def build_knowledge_source(
    vault: Path,
    *,
    config: RetrievalConfig | None = None,
    index_home: Path | None = None,
    defer_dense_model: bool = False,
) -> KnowledgeSource:
    return build_retrieval_source(
        vault,
        include_globs=("wiki/**/*.md",),
        config=config,
        index_home=index_home,
        defer_dense_model=defer_dense_model,
    )
