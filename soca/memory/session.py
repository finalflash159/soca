from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from soca.core.text_budget import truncate
from soca.memory.base import MemoryRole, MemoryTurn
from soca.memory.compaction_coordinator import CompactionResult, WorkingMemoryCompactionCoordinator
from soca.memory.summary import LocalSummaryWorkerProcess, build_production_summary_worker
from soca.memory.working import WorkingMemory, WorkingMemoryPolicy, approximate_tokens

RECENT_CONVERSATION_HEADER = "Recent conversation:"
VALID_ROLES = {"user", "assistant"}


@dataclass(frozen=True)
class SessionMemoryStats:
    current_tokens: int
    rendered_tokens: int
    hard_limit_tokens: int
    high_watermark_tokens: int
    target_tokens: int
    summary_tokens: int
    recent_tokens: int
    turn_count: int
    complete_turn_count: int
    summary_generation: int | None
    pending_compaction: bool
    worker_state: str


class SessionMemory:
    """Compatibility adapter over typed working-memory conversation turns.

    ``append(user)`` opens a turn and the following delivered ``append(assistant)``
    completes it.  The legacy flat ``turns`` view remains only for display and
    older integrations; compaction/state ownership lives in ``working``.
    """

    def __init__(
        self,
        turns: Iterable[MemoryTurn] | None = None,
        max_turns: int = 6,
        max_chars: int = 60_000,
        max_turn_chars: int = 500,
        *,
        thread_id: str = "default",
        summary_enabled: bool = True,
        summary_worker: LocalSummaryWorkerProcess | None = None,
        summary_model_root: Path | None = None,
        summary_threads: int | None = None,
        summary_gpu_layers: int = -1,
    ) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than 0")
        if max_chars <= len(RECENT_CONVERSATION_HEADER):
            raise ValueError("max_chars must leave room for the session memory header")
        if max_turn_chars <= 0:
            raise ValueError("max_turn_chars must be greater than 0")
        self.max_turns = max_turns
        self.max_chars = max_chars
        self.max_turn_chars = max_turn_chars
        self._summary_worker = (
            summary_worker
            if summary_worker is not None
            else (
                build_production_summary_worker(
                    model_root=summary_model_root,
                    n_threads=summary_threads,
                    n_gpu_layers=summary_gpu_layers,
                )
                if summary_enabled
                else None
            )
        )
        self.working = WorkingMemory(
            thread_id=thread_id,
            policy=WorkingMemoryPolicy(
                mode="background_summary" if summary_enabled else "trim_only"
            ),
        )
        self.compaction = WorkingMemoryCompactionCoordinator(
            self.working,
            self._summary_worker,
        )
        self._pending_sequences: list[int] = []
        # A completed async failure remains visible through the coordinator so
        # the UI can report it.  Its deterministic trim fallback, however,
        # must run only once: repeatedly trimming keeps the prompt below the
        # auto-compaction watermark forever and prevents a later retry.
        self._trimmed_failure_generation: int | None = None
        if turns is not None:
            for turn in turns:
                self.append(turn.role, turn.text)

    @property
    def turns(self) -> tuple[MemoryTurn, ...]:
        flattened: list[MemoryTurn] = []
        for turn in self.working.snapshot.turns:
            flattened.append(MemoryTurn("user", turn.user_text))
            if turn.assistant_text:
                flattened.append(MemoryTurn("assistant", turn.assistant_text))
        return tuple(flattened)

    def append(self, role: MemoryRole, text: str) -> None:
        if role not in VALID_ROLES:
            raise ValueError(f"Unsupported memory role: {role}")
        normalized = " ".join(text.strip().split())
        if not normalized:
            return
        self._poll_compaction()
        bounded = truncate(normalized, self.max_turn_chars)
        if role == "user":
            turn = self.working.begin_turn(bounded)
            self._pending_sequences.append(turn.sequence)
            self._enforce_hard_limit()
            return
        if not self._pending_sequences:
            # An assistant response without a user request has no trustworthy
            # turn ownership, therefore it cannot enter working memory.
            return
        sequence = self._pending_sequences.pop(0)
        self.working.finish_turn(sequence, bounded)
        self._enforce_hard_limit()
        if self.working.snapshot.token_count >= self.working.policy.high_watermark_tokens:
            result = self.compaction.request()
            if result.status in {"trim_only", "unavailable", "failed"}:
                self.working.trim_only()

    def clear(self) -> None:
        self.compaction.cancel()
        self.working = WorkingMemory(
            thread_id=self.working.thread_id,
            policy=self.working.policy,
        )
        self.compaction = WorkingMemoryCompactionCoordinator(
            self.working,
            self._summary_worker,
        )
        self._pending_sequences.clear()
        self._trimmed_failure_generation = None

    def render(self) -> str:
        self._poll_compaction()
        raw = self.working.render()
        if len(raw) <= self.max_chars:
            return raw
        marker = "\n\n" + RECENT_CONVERSATION_HEADER + "\n"
        if marker in raw:
            earlier, recent = raw.split(marker, maxsplit=1)
        elif raw.startswith(RECENT_CONVERSATION_HEADER + "\n"):
            earlier, recent = "", raw.removeprefix(RECENT_CONVERSATION_HEADER + "\n")
        else:
            return raw[: self.max_chars]
        prefix = earlier
        separator = "\n\n" if prefix else ""
        header = RECENT_CONVERSATION_HEADER
        used = len(prefix) + len(separator) + len(header)
        selected: list[str] = []
        for line in reversed(recent.splitlines()):
            cost = len(line) + 1
            if used + cost > self.max_chars:
                continue
            selected.append(line)
            used += cost
        if not selected:
            return prefix[: self.max_chars]
        recent_block = "\n".join([header, *reversed(selected)])
        return prefix + separator + recent_block

    def stats(self) -> SessionMemoryStats:
        snapshot = self.working.snapshot
        summary_section, recent_section = self.working.render_sections()
        return SessionMemoryStats(
            current_tokens=snapshot.token_count,
            rendered_tokens=approximate_tokens(self.render()),
            hard_limit_tokens=self.working.policy.hard_limit_tokens,
            high_watermark_tokens=self.working.policy.high_watermark_tokens,
            target_tokens=self.working.policy.target_tokens,
            summary_tokens=approximate_tokens(summary_section),
            recent_tokens=approximate_tokens(recent_section),
            turn_count=len(snapshot.turns),
            complete_turn_count=sum(turn.status == "complete" for turn in snapshot.turns),
            summary_generation=(
                snapshot.summary.generation if snapshot.summary is not None else None
            ),
            pending_compaction=snapshot.pending_compaction,
            worker_state=self.summary_worker_state,
        )

    def request_compaction(self) -> CompactionResult:
        current = self._poll_compaction()
        if current.status == "running":
            return current
        return self.compaction.request(manual=True)

    def compaction_status(self) -> CompactionResult:
        return self._poll_compaction()

    def cancel_compaction(self) -> CompactionResult:
        return self.compaction.cancel()

    def close(self) -> None:
        self.compaction.cancel()

    @property
    def summary_model_key(self) -> str | None:
        return self._summary_worker.spec.key if self._summary_worker is not None else None

    @property
    def summary_worker_state(self) -> str:
        return self._summary_worker.status.state if self._summary_worker is not None else "disabled"

    @property
    def summary_telemetry(self) -> dict[str, object] | None:
        return self.compaction.last_telemetry

    def _poll_compaction(self) -> CompactionResult:
        result = self.compaction.status()
        if (
            result.status == "failed"
            and result.generation != self._trimmed_failure_generation
        ):
            self.working.trim_only()
            self._trimmed_failure_generation = result.generation
        return result

    def _enforce_hard_limit(self) -> None:
        if self.working.snapshot.token_count <= self.working.policy.hard_limit_tokens:
            return
        self.compaction.cancel()
        self.working.trim_only()


__all__ = ["SessionMemory", "SessionMemoryStats"]
