from __future__ import annotations

import math
from dataclasses import dataclass

from soca.knowledge import KnowledgeDocument, KnowledgeHit, KnowledgeSource
from soca.memory.base import LongTermMemorySource, MemoryProfileResult
from soca.memory.episodes import EpisodeStore
from soca.memory.scoring import MemoryScoreConfig, rerank_memory_hits


@dataclass(frozen=True)
class CompositeMemoryConfig:
    top_k: int = 3
    candidate_limit: int = 12
    score: MemoryScoreConfig = MemoryScoreConfig()

    def __post_init__(self) -> None:
        if self.top_k < 1 or self.candidate_limit < self.top_k:
            raise ValueError("composite memory limits are invalid")


class CompositeMemorySource:
    """Merge profile retrieval and consented episode summaries without writes."""

    def __init__(
        self,
        profile_source: KnowledgeSource,
        profile_fallback: LongTermMemorySource,
        episodes: EpisodeStore,
        *,
        config: CompositeMemoryConfig | None = None,
    ) -> None:
        self._profile_source = profile_source
        self._profile_fallback = profile_fallback
        self._episodes = episodes
        self._config = config or CompositeMemoryConfig()

    def read_profile(self) -> str:
        return self._profile_fallback.read_profile()

    def retrieve_profile(self, query: str) -> MemoryProfileResult:
        normalized = " ".join(query.strip().split())
        if not normalized:
            return MemoryProfileResult(text=self.read_profile(), mode="blob")
        hits = list(self._profile_source.search(normalized, limit=self._config.candidate_limit))
        hits.extend(self._episode_hits(normalized))
        if not hits:
            return MemoryProfileResult(text="", mode="retrieved")
        ranked = rerank_memory_hits(
            tuple(hits),
            top_k=self._config.top_k,
            config=self._config.score,
        )
        text = "\n\n".join(
            f"[M{index}] {hit.document.path}\n{hit.snippet}"
            for index, hit in enumerate(ranked, start=1)
        )
        return MemoryProfileResult(text=text, hits=ranked, mode="retrieved")

    def _episode_hits(self, query: str) -> list[KnowledgeHit]:
        terms = set(query.casefold().split())
        if not terms:
            return []
        result: list[KnowledgeHit] = []
        for episode in self._episodes.load_all():
            text = " ".join((episode.summary, *episode.retained_facts)).strip()
            overlap = len(terms & set(text.casefold().split()))
            if overlap == 0:
                continue
            result.append(
                KnowledgeHit(
                    document=KnowledgeDocument(
                        id=f"episode:{episode.id}",
                        path=f"memory/episodes/{episode.id}.md",
                        title="Episode summary",
                        text=text,
                        tags=("episode",),
                    ),
                    score=float(overlap) / math.sqrt(max(1, len(terms))),
                    snippet=text,
                )
            )
        return result


__all__ = ["CompositeMemoryConfig", "CompositeMemorySource"]
