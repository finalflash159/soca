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
from soca.knowledge.relevance import RelevancePolicy
from soca.memory import CoreMemoryStore, MarkdownLongTermMemory, MemoryContextBuilder, SessionMemory
from soca.memory.retrieved import RetrievedMemory, RetrievedMemoryConfig
from soca.memory.scoring import MemoryScoreConfig

MemoryMode = Literal["blob", "retrieved"]


def default_memory_index_home() -> Path:
    return default_index_home() / "memory"


@dataclass(frozen=True)
class MemoryRuntimeConfig:
    mode: MemoryMode = "retrieved"
    top_k: int = 3
    context_chars: int = 64_000
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
    core: CoreMemoryStore | None
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
            core=None,
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
        effective_mode = str(getattr(source, "retrieval_mode", config.retrieval_mode))
        relevance_policy = RelevancePolicy.for_retrieval_mode(effective_mode)
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
            relevance_policy=relevance_policy,
        )
        status = f"enabled:retrieved:{effective_mode}"
    core = CoreMemoryStore(vault, max_chars=config.profile_chars)
    if not core.path.is_file():
        core_state = "empty"
    else:
        try:
            core.items()
            core_state = "ready"
        except (OSError, UnicodeError, ValueError):
            core_state = "degraded"
    return MemoryRuntimeSetup(
        builder=MemoryContextBuilder(
            long_term=long_term,
            session=session,
            core=core,
            max_chars=config.context_chars,
            profile_chars=config.profile_chars,
            relevance_policy=(
                relevance_policy
                if config.mode == "retrieved"
                else RelevancePolicy.for_retrieval_mode(config.retrieval_mode)
            ),
        ),
        long_term=long_term,
        core=core,
        status=status + f":core:{core_state}",
    )


__all__ = [
    "MemoryRuntimeConfig",
    "MemoryRuntimeSetup",
    "build_memory_runtime_setup",
    "default_memory_index_home",
]
