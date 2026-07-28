from __future__ import annotations

from soca.memory.working import WorkingMemory, WorkingSummaryArtifact


def _complete(memory: WorkingMemory, user: str, assistant: str) -> None:
    turn = memory.begin_turn(user)
    memory.finish_turn(turn.sequence, assistant)


def test_working_memory_compaction_uses_complete_turns_and_cas() -> None:
    memory = WorkingMemory(token_counter=lambda text: 1000 if "user 0" in text else 400)
    for index in range(6):
        _complete(memory, f"user {index}", f"assistant {index}")
    job = memory.prepare_compaction()
    assert job is not None
    assert len(job.frozen_turns) == 2
    artifact = WorkingSummaryArtifact(
        version=1,
        generation=job.generation,
        source_through_sequence=job.frozen_turns[-1].sequence,
        summary="Các quyết định trước đó vẫn cần được giữ.",
        decisions=("Giữ các quyết định trước đó.",),
    )
    assert memory.publish_summary(job, artifact) is True
    assert memory.publish_summary(job, artifact) is False
    snapshot = memory.snapshot
    assert snapshot.summary == artifact
    assert [turn.sequence for turn in snapshot.turns] == [3, 4, 5, 6]


def test_working_memory_does_not_include_undelivered_assistant_suffix() -> None:
    memory = WorkingMemory()
    turn = memory.begin_turn("Hãy giải thích RAG")
    memory.finish_turn(turn.sequence, "Câu đã phát", status="interrupted")
    assert "Câu đã phát" in memory.render()
    assert "suffix chưa phát" not in memory.render()


def test_working_memory_job_is_not_created_before_high_watermark() -> None:
    memory = WorkingMemory(token_counter=lambda _: 12)
    _complete(memory, "xin chào", "chào bạn")
    assert memory.prepare_compaction() is None
