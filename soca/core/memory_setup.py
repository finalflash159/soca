from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from soca.knowledge.factory import (
    DenseBackend,
    RetrievalConfig,
    RetrievalMode,
    build_retrieval_source,
)
from soca.knowledge.index.persistence import default_index_home
from soca.memory import MarkdownLongTermMemory, MemoryContextBuilder, SessionMemory
from soca.memory.retrieved import RetrievedMemory, RetrievedMemoryConfig
from soca.memory.scoring import MemoryScoreConfig

MemoryMode = Literal["blob", "retrieved"]


def default_memory_index_home() -> Path:
    return default_index_home() / "memory"


@dataclass(frozen=True)
class MemoryRuntimeConfig:
    mode: MemoryMode = "retrieved"
    top_k: int = 3
    context_chars: int = 2_200
    profile_chars: int = 900
    retrieval_mode: RetrievalMode = "chunk_sparse"
    dense_backend: DenseBackend = "fastembed"
    relevance_weight: float = 0.70
    recency_weight: float = 0.20
    importance_weight: float = 0.10
    recency_half_life_days: float = 30.0

    def __post_init__(self) -> None:
        if self.mode not in {"blob", "retrieved"}:
            raise ValueError("unknown memory mode")
        if self.retrieval_mode not in {"chunk_sparse", "hybrid"}:
            raise ValueError("unknown memory retrieval mode")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError("memory top_k must be positive")
        if self.context_chars < 1 or self.profile_chars < 1:
            raise ValueError("memory budgets must be positive")
        MemoryScoreConfig(
            relevance_weight=self.relevance_weight,
            recency_weight=self.recency_weight,
            importance_weight=self.importance_weight,
            recency_half_life_days=self.recency_half_life_days,
        )


@dataclass(frozen=True)
class MemoryRuntimeSetup:
    builder: MemoryContextBuilder
    long_term: MarkdownLongTermMemory | RetrievedMemory | None
    status: str


def build_memory_runtime_setup(
    vault: Path,
    *,
    session: SessionMemory,
    config: MemoryRuntimeConfig,
    index_home: Path | None = None,
) -> MemoryRuntimeSetup:
    if not vault.is_dir():
        return MemoryRuntimeSetup(
            builder=MemoryContextBuilder(
                long_term=None,
                session=session,
                max_chars=config.context_chars,
                profile_chars=config.profile_chars,
            ),
            long_term=None,
            status=f"session-only:vault_missing:{vault}",
        )

    blob = MarkdownLongTermMemory(vault, max_chars=config.profile_chars)
    if config.mode == "blob":
        long_term: MarkdownLongTermMemory | RetrievedMemory = blob
        status = "enabled:blob"
    else:
        source = build_retrieval_source(
            vault,
            include_globs=("memory/**/*.md",),
            config=RetrievalConfig(
                mode=config.retrieval_mode,
                dense_backend=config.dense_backend,
            ),
            index_home=index_home or default_memory_index_home(),
        )
        long_term = RetrievedMemory(
            source,
            blob,
            config=RetrievedMemoryConfig(
                top_k=config.top_k,
                max_chars=config.profile_chars,
                score=MemoryScoreConfig(
                    relevance_weight=config.relevance_weight,
                    recency_weight=config.recency_weight,
                    importance_weight=config.importance_weight,
                    recency_half_life_days=config.recency_half_life_days,
                ),
            ),
        )
        status = f"enabled:retrieved:{config.retrieval_mode}"
    return MemoryRuntimeSetup(
        builder=MemoryContextBuilder(
            long_term=long_term,
            session=session,
            max_chars=config.context_chars,
            profile_chars=config.profile_chars,
        ),
        long_term=long_term,
        status=status,
    )


__all__ = [
    "MemoryRuntimeConfig",
    "MemoryRuntimeSetup",
    "build_memory_runtime_setup",
    "default_memory_index_home",
]
