from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from soca.knowledge.base import KnowledgeSource
from soca.knowledge.cached_source import CachedMarkdownVaultKnowledgeSource
from soca.knowledge.hybrid_source import HybridConfig, HybridKnowledgeSource
from soca.knowledge.retrievers.dense import FastEmbedModel, Model2VecModel

LOGGER = logging.getLogger(__name__)
RetrievalMode = Literal["cached_sparse", "chunk_sparse", "hybrid"]
DenseBackend = Literal["fastembed", "model2vec"]
IndexLifecycle = Literal["legacy", "v2"]


@dataclass(frozen=True)
class RetrievalConfig:
    mode: RetrievalMode = "cached_sparse"
    dense_backend: DenseBackend = "fastembed"
    rrf_k: int = 60
    per_retriever_limit: int = 12
    lifecycle: IndexLifecycle = "v2"

    def __post_init__(self) -> None:
        if self.mode not in {"cached_sparse", "chunk_sparse", "hybrid"}:
            raise ValueError("unknown retrieval mode")
        if self.dense_backend not in {"fastembed", "model2vec"}:
            raise ValueError("unknown dense backend")
        if self.lifecycle not in {"legacy", "v2"}:
            raise ValueError("unknown index lifecycle")
        if (
            isinstance(self.rrf_k, bool)
            or not isinstance(self.rrf_k, int)
            or self.rrf_k < 1
            or isinstance(self.per_retriever_limit, bool)
            or not isinstance(self.per_retriever_limit, int)
            or self.per_retriever_limit < 1
        ):
            raise ValueError("retrieval limits must be positive")


def _build_model(backend: DenseBackend):
    return FastEmbedModel() if backend == "fastembed" else Model2VecModel()


def build_retrieval_source(
    vault: Path,
    *,
    include_globs: tuple[str, ...],
    config: RetrievalConfig | None = None,
    index_home: Path | None = None,
) -> KnowledgeSource:
    resolved = config or RetrievalConfig()
    corpus_kind = "memory" if include_globs == ("memory/**/*.md",) else "knowledge"
    common = {
        "index_home": index_home,
        "include_globs": include_globs,
        "lifecycle": resolved.lifecycle,
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
            ),
            **common,
        )

    try:
        model = _build_model(resolved.dense_backend)
    except (ImportError, OSError, RuntimeError, ValueError):
        LOGGER.warning("Dense retrieval unavailable; using cached sparse", exc_info=True)
        return build_retrieval_source(
            vault,
            include_globs=include_globs,
            config=replace(resolved, mode="cached_sparse"),
            index_home=index_home,
        )

    return HybridKnowledgeSource(
        vault,
        model=model,
        config=HybridConfig(
            rrf_k=resolved.rrf_k,
            per_retriever_limit=resolved.per_retriever_limit,
        ),
        **common,
    )


def build_knowledge_source(
    vault: Path,
    *,
    config: RetrievalConfig | None = None,
    index_home: Path | None = None,
) -> KnowledgeSource:
    return build_retrieval_source(
        vault,
        include_globs=("wiki/**/*.md",),
        config=config,
        index_home=index_home,
    )
