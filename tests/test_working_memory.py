from __future__ import annotations

from soca.memory.working import WorkingMemory, WorkingSummaryArtifact


def _complete(memory: WorkingMemory, user: str, assistant: str) -> None:
    turn = memory.begin_turn(user)
    memory.finish_turn(turn.sequence, assistant)


def test_working_memory_compaction_uses_complete_turns_and_cas() -> None:
    memory = WorkingMemory(token_counter=lambda text: 15_000 if "user 0" in text else 400)
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
    rendered = memory.render()
    assert "Earlier conversation state:" in rendered
    assert "Active decisions:" in rendered
    assert "Giữ các quyết định trước đó." in rendered


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


def test_manual_compaction_requires_five_complete_turns() -> None:
    memory = WorkingMemory()
    for index in range(4):
        _complete(memory, f"user {index}", f"assistant {index}")

    assert memory.policy.manual_compaction_minimum_complete_turns == 5
    assert memory.prepare_compaction(force=True) is None

    _complete(memory, "user 4", "assistant 4")
    job = memory.prepare_compaction(force=True)

    assert job is not None
    assert [turn.sequence for turn in job.frozen_turns] == [1]


def test_working_summary_budget_covers_structured_fields() -> None:
    try:
        WorkingSummaryArtifact(
            version=1,
            generation=1,
            source_through_sequence=1,
            summary="",
            open_items=("việc rất dài " * 300,),
        )
    except ValueError as exc:
        assert "256-token content budget" in str(exc)
    else:
        raise AssertionError("oversized structured state must be rejected")
