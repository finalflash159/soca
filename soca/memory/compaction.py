from __future__ import annotations

import logging
import re
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, Full, Queue

from soca.core.text_budget import truncate
from soca.memory.base import MemoryRole, MemoryTurn

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompactionConfig:
    recent_turns: int = 8
    summary_chars: int = 1_600
    queue_size: int = 4
    flush_timeout_seconds: float = 2.0
    llm_summary_enabled: bool = False

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (self.recent_turns, self.summary_chars, self.queue_size)
        ):
            raise ValueError("compaction integer limits must be positive")
        if self.flush_timeout_seconds <= 0:
            raise ValueError("compaction flush timeout must be positive")
        if not isinstance(self.llm_summary_enabled, bool):
            raise ValueError("llm_summary_enabled must be a boolean")


@dataclass(frozen=True)
class WorkingMemorySnapshot:
    summary: str
    recent_turns: tuple[MemoryTurn, ...]
    compacted_turn_count: int
    generation: int


class WorkingMemory:
    def __init__(
        self,
        *,
        config: CompactionConfig | None = None,
        summarizer: Callable[[tuple[MemoryTurn, ...]], str] | None = None,
    ) -> None:
        self.config = config or CompactionConfig()
        self._summarizer = summarizer
        self._turns: deque[MemoryTurn] = deque()
        self._snapshot = WorkingMemorySnapshot("", (), 0, 0)
        self._lock = threading.RLock()
        self._queue: Queue[tuple[int, tuple[MemoryTurn, ...]]] = Queue(self.config.queue_size)
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        if self.config.llm_summary_enabled and self._summarizer is not None:
            self._worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker.start()

    @property
    def snapshot(self) -> WorkingMemorySnapshot:
        with self._lock:
            current = self._snapshot
            return WorkingMemorySnapshot(
                current.summary,
                tuple(current.recent_turns),
                current.compacted_turn_count,
                current.generation,
            )

    def append(self, role: MemoryRole, text: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("unsupported memory role")
        normalized = " ".join(text.strip().split())
        if not normalized:
            return
        with self._lock:
            self._turns.append(MemoryTurn(role, normalized))
            if len(self._turns) <= self.config.recent_turns:
                self._publish(tuple(self._turns), self._snapshot.summary, self._snapshot.compacted_turn_count)
                return
            older_count = len(self._turns) - self.config.recent_turns
            older = tuple(list(self._turns)[:older_count])
            recent = tuple(list(self._turns)[older_count:])
            summary = _extractive_summary(older, self.config.summary_chars)
            generation = self._snapshot.generation + 1
            self._turns = deque(recent)
            self._publish(recent, summary, self._snapshot.compacted_turn_count + older_count)
            if self._worker is not None:
                try:
                    self._queue.put_nowait((generation, older))
                except Full:
                    pass

    def render(self) -> str:
        current = self.snapshot
        parts: list[str] = []
        if current.summary:
            parts.append(f"Earlier conversation summary:\n{current.summary}")
        if current.recent_turns:
            lines = ["Recent conversation:"]
            lines.extend(
                f"{'User' if turn.role == 'user' else 'Assistant'}: {turn.text}"
                for turn in current.recent_turns
            )
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def flush(self, timeout: float | None = None) -> None:
        if self._worker is None:
            return
        wait_seconds = timeout if timeout is not None else self.config.flush_timeout_seconds
        end = threading.Event()
        while not self._queue.empty() and wait_seconds > 0:
            end.wait(min(0.01, wait_seconds))
            wait_seconds -= 0.01

    def close(self, timeout: float | None = None) -> None:
        self.flush(timeout)
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=timeout or self.config.flush_timeout_seconds)
            self._worker = None

    def _publish(self, recent: tuple[MemoryTurn, ...], summary: str, count: int) -> None:
        self._snapshot = WorkingMemorySnapshot(
            summary=summary,
            recent_turns=tuple(recent),
            compacted_turn_count=count,
            generation=self._snapshot.generation + (1 if count else 0),
        )

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                generation, turns = self._queue.get(timeout=0.05)
            except Empty:
                continue
            try:
                assert self._summarizer is not None
                summary = truncate(self._summarizer(turns), self.config.summary_chars)
                with self._lock:
                    if generation == self._snapshot.generation and summary:
                        self._snapshot = WorkingMemorySnapshot(
                            summary,
                            self._snapshot.recent_turns,
                            self._snapshot.compacted_turn_count,
                            self._snapshot.generation,
                        )
            except Exception as exc:  # noqa: BLE001 - background refinement is best effort
                LOGGER.warning("Memory summary refinement failed (%s)", type(exc).__name__)
            finally:
                self._queue.task_done()


def _extractive_summary(turns: tuple[MemoryTurn, ...], max_chars: int) -> str:
    selected: list[str] = []
    seen: set[str] = set()
    priority = re.compile(r"(quyết|chọn|sửa|đính chính|thích|prefer|todo|cần|http|/|\d)", re.I)
    for turn in reversed(turns):
        line = f"{'User' if turn.role == 'user' else 'Assistant'}: {turn.text}"
        key = " ".join(line.lower().split())
        if key in seen:
            continue
        seen.add(key)
        if priority.search(line) or not selected:
            selected.append(line)
    selected.reverse()
    return truncate("\n".join(selected), max_chars)


__all__ = ["CompactionConfig", "WorkingMemory", "WorkingMemorySnapshot"]
