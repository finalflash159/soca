from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from soca.knowledge.base import KnowledgeSource
from soca.knowledge.cached_source import CachedMarkdownVaultKnowledgeSource
from soca.knowledge.hybrid_source import HybridConfig, HybridKnowledgeSource
from soca.knowledge.indexing.models import load_model

RetrievalMode = Literal["cached_sparse", "chunk_sparse", "hybrid"]
DenseBackend = Literal["aiteamvn_v2"]
IndexLifecycle = Literal["legacy", "v2"]


@dataclass(frozen=True)
class RetrievalConfig:
    mode: RetrievalMode = "hybrid"
    dense_backend: str = "aiteamvn_v2"
    rrf_k: int = 60
    per_retriever_limit: int = 12
    lifecycle: IndexLifecycle = "v2"
    watcher_enabled: bool = True
    watcher_interval_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.mode not in {"cached_sparse", "chunk_sparse", "hybrid"}:
            raise ValueError("unknown retrieval mode")
        if self.mode == "hybrid" and self.dense_backend != "aiteamvn_v2":
            raise ValueError("unknown dense backend")
        if self.lifecycle not in {"legacy", "v2"}:
            raise ValueError("unknown index lifecycle")
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

    model = _build_model(resolved.dense_backend)

    source = HybridKnowledgeSource(
        vault,
        model=model,
        config=HybridConfig(
            rrf_k=resolved.rrf_k,
            per_retriever_limit=resolved.per_retriever_limit,
            sparse_backend="bm25",
            fusion="linear",
            dense_weight=0.75,
        ),
        **common,
    )
    if resolved.lifecycle == "v2" and resolved.watcher_enabled:
        source.activate_watcher(interval_seconds=resolved.watcher_interval_seconds)
    return source


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
