from __future__ import annotations

from soca.memory.compaction_coordinator import WorkingMemoryCompactionCoordinator
from soca.memory.working import WorkingMemory


def test_manual_and_auto_use_same_coordinator_and_do_not_fake_summary() -> None:
    memory = WorkingMemory(token_counter=lambda _: 1000)
    for index in range(6):
        turn = memory.begin_turn(f"user {index}")
        memory.finish_turn(turn.sequence, f"assistant {index}")
    coordinator = WorkingMemoryCompactionCoordinator(memory)
    result = coordinator.request(manual=True)
    assert result.status == "trim_only"
    assert memory.snapshot.pending_compaction is False
    assert coordinator.status().status == "idle"
