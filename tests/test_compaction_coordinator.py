from __future__ import annotations

from typing import Any, cast

from soca.memory.compaction_coordinator import WorkingMemoryCompactionCoordinator
from soca.memory.summary import LocalSummaryWorkerProcess
from soca.memory.working import (
    CompactionJob,
    WorkingMemory,
    WorkingMemoryPolicy,
    WorkingSummaryArtifact,
)


class _ImmediateWorker:
    def __init__(self, *, empty: bool = False) -> None:
        self.job: CompactionJob | None = None
        self.empty = empty

    def start(self, job: CompactionJob) -> bool:
        self.job = job
        return True

    def poll(self) -> dict[str, object] | None:
        if self.job is None:
            return None
        job = self.job
        self.job = None
        return {
            "ok": True,
            "artifact": WorkingSummaryArtifact(
                version=1,
                generation=job.generation,
                source_through_sequence=job.frozen_turns[-1].sequence,
                summary="" if self.empty else "Đã compact lịch sử cũ.",
            ).to_dict(),
            "latency_ms": 12.5,
        }

    def cancel(self) -> bool:
        self.job = None
        return True


def test_manual_and_auto_use_same_coordinator_and_do_not_fake_summary() -> None:
    memory = WorkingMemory(token_counter=lambda _: 15_000)
    for index in range(6):
        turn = memory.begin_turn(f"user {index}")
        memory.finish_turn(turn.sequence, f"assistant {index}")
    coordinator = WorkingMemoryCompactionCoordinator(memory)
    result = coordinator.request(manual=True)
    assert result.status == "trim_only"
    assert memory.snapshot.pending_compaction is False
    assert coordinator.status().status == "trim_only"


def test_background_summary_without_worker_is_unavailable_not_trimmed() -> None:
    memory = WorkingMemory(
        token_counter=lambda _: 15_000,
        policy=WorkingMemoryPolicy(mode="background_summary"),
    )
    for index in range(6):
        turn = memory.begin_turn(f"user {index}")
        memory.finish_turn(turn.sequence, f"assistant {index}")
    coordinator = WorkingMemoryCompactionCoordinator(memory)

    result = coordinator.request()

    assert result.status == "unavailable"
    assert result.detail == "summary_worker_not_configured"
    assert len(memory.snapshot.turns) == 6


def test_manual_compaction_reports_before_and_after_tokens() -> None:
    memory = WorkingMemory()
    for index in range(6):
        turn = memory.begin_turn(f"user {index} " + "nội dung " * 20)
        memory.finish_turn(turn.sequence, f"assistant {index} " + "phản hồi " * 20)
    worker = cast(LocalSummaryWorkerProcess, cast(Any, _ImmediateWorker()))
    coordinator = WorkingMemoryCompactionCoordinator(memory, worker)

    accepted = coordinator.request(manual=True)
    published = coordinator.status()

    assert accepted.status == "accepted"
    assert accepted.before_tokens is not None
    assert accepted.compacted_turns == 4
    assert accepted.complete_turns == 6
    assert accepted.minimum_complete_turns == 5
    assert published.status == "published"
    assert published.before_tokens == accepted.before_tokens
    assert published.after_tokens == memory.snapshot.token_count
    assert published.compacted_turns == 4
    assert published.elapsed_ms is not None
    assert coordinator.status() == published
    assert memory.snapshot.summary is not None


def test_manual_compaction_noop_reports_five_turn_requirement() -> None:
    memory = WorkingMemory()
    for index in range(4):
        turn = memory.begin_turn(f"user {index}")
        memory.finish_turn(turn.sequence, f"assistant {index}")
    coordinator = WorkingMemoryCompactionCoordinator(memory)

    result = coordinator.request(manual=True)

    assert result.status == "noop"
    assert result.detail == "not_enough_complete_turns"
    assert result.complete_turns == 4
    assert result.minimum_complete_turns == 5


def test_empty_first_summary_preserves_original_turns() -> None:
    memory = WorkingMemory()
    for index in range(5):
        turn = memory.begin_turn(f"user {index}")
        memory.finish_turn(turn.sequence, f"assistant {index}")
    worker = cast(
        LocalSummaryWorkerProcess,
        cast(Any, _ImmediateWorker(empty=True)),
    )
    coordinator = WorkingMemoryCompactionCoordinator(memory, worker)

    accepted = coordinator.request(manual=True)
    failed = coordinator.status()

    assert accepted.status == "accepted"
    assert failed.status == "failed"
    assert failed.detail == "empty_continuity_summary"
    assert len(memory.snapshot.turns) == 5
    assert memory.snapshot.summary is None


def test_empty_rolling_summary_cannot_drop_previous_state() -> None:
    memory = WorkingMemory()
    for index in range(5):
        turn = memory.begin_turn(f"user {index}")
        memory.finish_turn(turn.sequence, f"assistant {index}")
    initial_job = memory.prepare_compaction(force=True)
    assert initial_job is not None
    assert memory.publish_summary(
        initial_job,
        WorkingSummaryArtifact(
            version=1,
            generation=initial_job.generation,
            source_through_sequence=initial_job.frozen_turns[-1].sequence,
            summary="Quyết định đang hoạt động: dùng TTS local.",
        ),
    )
    for index in range(3):
        next_turn = memory.begin_turn(f"user next {index}")
        memory.finish_turn(next_turn.sequence, f"assistant next {index}")
    worker = cast(
        LocalSummaryWorkerProcess,
        cast(Any, _ImmediateWorker(empty=True)),
    )
    coordinator = WorkingMemoryCompactionCoordinator(memory, worker)

    accepted = coordinator.request(manual=True)
    failed = coordinator.status()

    assert accepted.status == "accepted"
    assert failed.status == "failed"
    assert failed.detail == "empty_continuity_summary"
    assert memory.snapshot.summary is not None
    assert "TTS local" in memory.snapshot.summary.render()
    assert len(memory.snapshot.turns) == 5
