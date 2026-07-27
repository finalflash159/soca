from __future__ import annotations

from soca.memory.compaction import CompactionConfig, WorkingMemory


def test_working_memory_compacts_without_exceeding_budget() -> None:
    memory = WorkingMemory(config=CompactionConfig(recent_turns=2, summary_chars=80))
    for index in range(5):
        memory.append("user", f"decision {index} keeps path /tmp/item-{index}")
    snapshot = memory.snapshot
    assert len(snapshot.recent_turns) == 2
    assert len(snapshot.summary) <= 80
    assert snapshot.compacted_turn_count == 3
    memory.close()


def test_summary_refinement_is_background() -> None:
    calls: list[int] = []

    def summarizer(turns):
        calls.append(len(turns))
        return "refined"

    memory = WorkingMemory(
        config=CompactionConfig(recent_turns=1, summary_chars=40, llm_summary_enabled=True),
        summarizer=summarizer,
    )
    memory.append("user", "keep this fact 1")
    memory.append("assistant", "keep this fact 2")
    memory.flush(1.0)
    memory.close()
    assert calls
