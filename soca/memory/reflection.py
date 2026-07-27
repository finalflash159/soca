from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from soca.memory.episodes import MemoryEpisode
from soca.memory.proposals import MemoryProposal, ProposalKind, ProposalStore

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReflectionConfig:
    max_proposals: int = 8
    statement_chars: int = 1_000
    evidence_chars: int = 500

    def __post_init__(self) -> None:
        if self.max_proposals < 1 or self.statement_chars < 1 or self.evidence_chars < 1:
            raise ValueError("reflection limits must be positive")


class ReflectionService:
    def __init__(
        self,
        store: ProposalStore,
        *,
        config: ReflectionConfig | None = None,
    ) -> None:
        self.store = store
        self.config = config or ReflectionConfig()

    def reflect(
        self,
        episode: MemoryEpisode,
        generator: Callable[[MemoryEpisode], Iterable[dict[str, object]]],
    ) -> tuple[MemoryProposal, ...]:
        try:
            raw_items = tuple(generator(episode))[: self.config.max_proposals]
        except Exception as exc:  # noqa: BLE001 - reflection is background best effort
            LOGGER.warning("Memory reflection failed (%s)", type(exc).__name__)
            return ()
        proposals: list[MemoryProposal] = []
        for raw in raw_items:
            try:
                proposal = self._build_proposal(episode, raw)
                if any(
                    existing.statement.casefold() == proposal.statement.casefold()
                    and existing.source_episode_id == proposal.source_episode_id
                    for existing in self.store.list()
                ):
                    continue
                proposals.append(self.store.put(proposal))
            except (TypeError, ValueError):
                continue
        return tuple(proposals)

    def _build_proposal(self, episode: MemoryEpisode, raw: dict[str, object]) -> MemoryProposal:
        if not isinstance(raw, dict):
            raise ValueError("proposal candidate must be an object")
        return MemoryProposal(
            id=str(uuid4()),
            kind=cast(ProposalKind, raw["kind"]),
            statement=str(raw["statement"])[: self.config.statement_chars],
            evidence_excerpt=str(raw["evidence_excerpt"])[: self.config.evidence_chars],
            confidence=float(cast(str | float | int, raw["confidence"])),
            source_episode_id=episode.id,
            created_at=datetime.now(UTC),
        )


class BackgroundReflection:
    def __init__(self, service: ReflectionService) -> None:
        self._service = service
        self._jobs: list[tuple[MemoryEpisode, Callable[[MemoryEpisode], Iterable[dict[str, object]]]]] = []
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._stop = threading.Event()
        self._worker.start()

    def submit(
        self,
        episode: MemoryEpisode,
        generator: Callable[[MemoryEpisode], Iterable[dict[str, object]]],
    ) -> None:
        with self._lock:
            self._jobs.append((episode, generator))

    def close(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._worker.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                job = self._jobs.pop(0) if self._jobs else None
            if job is None:
                self._stop.wait(0.05)
                continue
            self._service.reflect(*job)


__all__ = ["BackgroundReflection", "ReflectionConfig", "ReflectionService"]
