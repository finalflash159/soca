from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

from soca.knowledge import KnowledgeHit
from soca.memory.frontmatter import MemoryMetadata, parse_memory_frontmatter


@dataclass(frozen=True)
class MemoryScoreConfig:
    relevance_weight: float = 0.70
    recency_weight: float = 0.20
    importance_weight: float = 0.10
    recency_half_life_days: float = 30.0

    def __post_init__(self) -> None:
        weights = (self.relevance_weight, self.recency_weight, self.importance_weight)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in weights):
            raise ValueError("memory score weights must be numeric")
        if any(not math.isfinite(float(value)) or not 0.0 <= value <= 1.0 for value in weights):
            raise ValueError("memory score weights must be between 0 and 1")
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("memory score weights must sum to 1")
        if (
            isinstance(self.recency_half_life_days, bool)
            or not isinstance(self.recency_half_life_days, (int, float))
            or not math.isfinite(float(self.recency_half_life_days))
            or self.recency_half_life_days <= 0
        ):
            raise ValueError("memory recency half-life must be positive")


@dataclass(frozen=True)
class MemoryScore:
    relevance: float
    recency: float
    importance: float
    total: float


@dataclass(frozen=True)
class MemoryHit:
    knowledge_hit: KnowledgeHit
    score: MemoryScore

    @property
    def document(self):
        return self.knowledge_hit.document

    @property
    def snippet(self) -> str:
        return self.knowledge_hit.snippet

    @property
    def line_start(self) -> int | None:
        return self.knowledge_hit.line_start

    @property
    def line_end(self) -> int | None:
        return self.knowledge_hit.line_end


def rerank_memory_hits(
    hits: tuple[KnowledgeHit, ...],
    *,
    top_k: int,
    config: MemoryScoreConfig | None = None,
    now: datetime | None = None,
) -> tuple[MemoryHit, ...]:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("memory top_k must be positive")
    resolved = config or MemoryScoreConfig()
    current = (now or datetime.now(UTC)).astimezone(UTC)
    scored: list[MemoryHit] = []
    for rank, hit in enumerate(hits, start=1):
        metadata = _metadata_for(hit)
        relevance = 1.0 / rank
        recency = _recency(metadata, current, resolved.recency_half_life_days)
        importance = (metadata.importance if metadata.importance is not None else 5) / 10.0
        total = (
            resolved.relevance_weight * relevance
            + resolved.recency_weight * recency
            + resolved.importance_weight * importance
        )
        scored.append(MemoryHit(hit, MemoryScore(relevance, recency, importance, total)))
    scored.sort(key=lambda item: (-item.score.total, -item.score.relevance, item.document.path, item.knowledge_hit.document.id))
    return tuple(scored[:top_k])


def _metadata_for(hit: KnowledgeHit) -> MemoryMetadata:
    try:
        return parse_memory_frontmatter(hit.document.text)
    except (ValueError, UnicodeError):
        return MemoryMetadata()


def _recency(metadata: MemoryMetadata, now: datetime, half_life_days: float) -> float:
    timestamp = metadata.updated_at or metadata.created_at
    if timestamp is None:
        return 0.0
    age_days = max(0.0, (now - timestamp.astimezone(UTC)).total_seconds() / 86_400)
    return 2.0 ** (-age_days / half_life_days)


__all__ = ["MemoryHit", "MemoryScore", "MemoryScoreConfig", "rerank_memory_hits"]
