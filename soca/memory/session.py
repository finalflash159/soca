from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from soca.core.text_budget import truncate
from soca.memory.base import MemoryRole, MemoryTurn
from soca.memory.compaction_coordinator import CompactionResult, WorkingMemoryCompactionCoordinator
from soca.memory.session_store import (
    SessionCheckpointStore,
    _payload_digest,
)
from soca.memory.summary import LocalSummaryWorkerProcess, build_production_summary_worker
from soca.memory.working import WorkingMemory, WorkingMemoryPolicy, approximate_tokens


class MemoryCapacityError(RuntimeError):
    """Working memory exceeded its contract without a safe summary."""


RECENT_CONVERSATION_HEADER = "Recent conversation:"
VALID_ROLES = {"user", "assistant"}
SessionPersistence = Literal["ram_only", "local_resumable"]


@dataclass(frozen=True)
class SessionMemoryStats:
    thread_id: str
    persistence: SessionPersistence
    checkpoint_enabled: bool
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
    completes it.  The flat ``turns`` view remains only for display and older
    integrations; compaction/state ownership lives in ``working``.
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
        persistence: SessionPersistence = "ram_only",
        checkpoint_store: SessionCheckpointStore | None = None,
        resume: bool = False,
        working_policy: WorkingMemoryPolicy | None = None,
    ) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than 0")
        if max_chars <= len(RECENT_CONVERSATION_HEADER):
            raise ValueError("max_chars must leave room for the session memory header")
        if max_turn_chars <= 0:
            raise ValueError("max_turn_chars must be greater than 0")
        if persistence not in {"ram_only", "local_resumable"}:
            raise ValueError("unknown session persistence mode")
        if persistence == "local_resumable" and checkpoint_store is None:
            raise ValueError("local_resumable sessions require a checkpoint store")
        if resume and persistence != "local_resumable":
            raise ValueError("resume requires local_resumable persistence")
        self.max_turns = max_turns
        self.max_chars = max_chars
        self.max_turn_chars = max_turn_chars
        self.persistence: SessionPersistence = persistence
        self.checkpoint_store = checkpoint_store
        self._checkpoint_revision: int | None = None
        self._checkpoint_digest: str | None = None
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
            policy=working_policy
            or WorkingMemoryPolicy(mode="background_summary" if summary_enabled else "trim_only"),
        )
        if resume and checkpoint_store is not None:
            loaded, revision, digest = checkpoint_store.load_with_metadata(
                thread_id,
                policy=self.working.policy,
            )
            if loaded is not None:
                self.working = loaded
                self._checkpoint_revision = revision
                self._checkpoint_digest = digest
        self.compaction = WorkingMemoryCompactionCoordinator(
            self.working,
            self._summary_worker,
        )
        self._pending_sequences: list[int] = []
        self._closed = False
        self._pending_sequences.extend(
            turn.sequence for turn in self.working.snapshot.turns if turn.status == "pending"
        )
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
            self._save_checkpoint()
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
            if result.status == "trim_only" and self.working.policy.mode == "trim_only":
                self.working.trim_only()
        self._save_checkpoint()

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
        self._delete_checkpoint()

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
            thread_id=snapshot.thread_id,
            persistence=self.persistence,
            checkpoint_enabled=self.checkpoint_path is not None,
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
        if self._closed:
            return
        failures: list[tuple[str, Exception]] = []
        try:
            self.compaction.cancel()
        except Exception as exc:  # noqa: BLE001 - continue worker teardown
            failures.append(("compaction", exc))
        close_worker = getattr(self._summary_worker, "close", None)
        if callable(close_worker):
            try:
                close_worker()
            except Exception as exc:  # noqa: BLE001 - continue worker teardown
                failures.append(("summary_worker", exc))
        try:
            self._save_checkpoint()
        except Exception as exc:  # noqa: BLE001 - expose checkpoint failure
            failures.append(("checkpoint", exc))
        if failures:
            details = "; ".join(f"{name}: {error}" for name, error in failures)
            raise RuntimeError(f"Session memory cleanup failed: {details}") from failures[0][1]
        self._closed = True

    @property
    def summary_model_key(self) -> str | None:
        return self._summary_worker.spec.key if self._summary_worker is not None else None

    @property
    def summary_worker_state(self) -> str:
        return self._summary_worker.status.state if self._summary_worker is not None else "disabled"

    @property
    def summary_telemetry(self) -> dict[str, object] | None:
        return self.compaction.last_telemetry

    @property
    def checkpoint_path(self) -> Path | None:
        if self.persistence != "local_resumable" or self.checkpoint_store is None:
            return None
        return self.checkpoint_store._path(self.working.thread_id)

    def _poll_compaction(self) -> CompactionResult:
        result = self.compaction.status()
        if result.status in {"published", "trim_only", "unavailable", "failed"}:
            self._save_checkpoint()
        return result

    def _enforce_hard_limit(self) -> None:
        if self.working.snapshot.token_count <= self.working.policy.hard_limit_tokens:
            return
        self.compaction.cancel()
        if self.working.policy.mode == "trim_only":
            self.working.trim_only()
            return
        raise MemoryCapacityError(
            "working memory exceeded its hard limit before a valid summary was published"
        )

    def _save_checkpoint(self) -> None:
        if self.persistence != "local_resumable" or self.checkpoint_store is None:
            return
        self.checkpoint_store.save(
            self.working,
            expected_revision=self._checkpoint_revision,
            expected_digest=self._checkpoint_digest,
        )
        self._checkpoint_revision = self.working.snapshot.revision
        self._checkpoint_digest = _payload_digest(self.working.to_dict())

    def _delete_checkpoint(self) -> None:
        if (
            self.persistence == "local_resumable"
            and self.checkpoint_store is not None
            and self._checkpoint_revision is not None
        ):
            self.checkpoint_store.delete(
                self.working.thread_id,
                expected_revision=self._checkpoint_revision,
                expected_digest=self._checkpoint_digest,
            )
            self._checkpoint_revision = None
            self._checkpoint_digest = None


__all__ = [
    "MemoryCapacityError",
    "SessionMemory",
    "SessionMemoryStats",
    "SessionPersistence",
]
