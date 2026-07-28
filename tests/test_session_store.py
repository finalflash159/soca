from __future__ import annotations

import stat
from pathlib import Path

import pytest

from soca.memory.session_store import SessionCheckpointStore
from soca.memory.working import WorkingMemory


def test_checkpoint_round_trip_is_atomic_and_private(tmp_path: Path) -> None:
    memory = WorkingMemory(thread_id="chat-1")
    turn = memory.begin_turn("xin chào")
    memory.finish_turn(turn.sequence, "chào bạn")
    store = SessionCheckpointStore(tmp_path / "sessions")
    path = store.save(memory)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    loaded = store.load("chat-1")
    assert loaded is not None
    assert loaded.snapshot.turns == memory.snapshot.turns
    assert store.delete("chat-1") is True
    assert store.load("chat-1") is None


def test_checkpoint_rejects_non_private_file(tmp_path: Path) -> None:
    store = SessionCheckpointStore(tmp_path / "sessions")
    path = store._path("chat-1")
    path.write_text('{"version":1,"thread_id":"chat-1","turns":[],"generation":0,"revision":0,"summary":null}')
    path.chmod(0o644)
    with pytest.raises(ValueError, match="permissions"):
        store.load("chat-1")
