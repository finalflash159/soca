from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from soca.memory.commands import MemoryCommands
from soca.memory.proposals import MemoryProposal, ProposalStore


def test_approve_is_idempotent_and_writes_a_safe_note(tmp_path) -> None:
    proposal_id = str(uuid4())
    episode_id = str(uuid4())
    proposal = MemoryProposal(
        proposal_id,
        "preference",
        "The user prefers concise explanations",
        "A validated session summary",
        0.9,
        episode_id,
        datetime.now(UTC),
    )
    store = ProposalStore(tmp_path / "proposals")
    store.put(proposal)
    commands = MemoryCommands(tmp_path / "vault", store)
    first = commands.approve(proposal_id)
    second = commands.approve(proposal_id)
    assert first.status == second.status == "approved"
    assert first.note_path
    assert len(list((tmp_path / "vault" / "memory" / "captured").glob("*.md"))) == 1
