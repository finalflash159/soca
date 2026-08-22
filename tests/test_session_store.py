from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

import soca.memory.session_store as session_store
from soca.memory.session_store import SessionCheckpointStore
from soca.memory.working import WorkingMemory


def test_checkpoint_round_trip_is_atomic_and_private(tmp_path: Path) -> None:
    memory = WorkingMemory(thread_id="chat-1")
    turn = memory.begin_turn("xin chào")
    memory.finish_turn(turn.sequence, "chào bạn")
    store = SessionCheckpointStore(tmp_path / "sessions")
    path = store.save(memory)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["persistence"] == "local_resumable"
    loaded = store.load("chat-1")
    assert loaded is not None
    assert loaded.snapshot.turns == memory.snapshot.turns
    assert store.delete("chat-1") is True
    assert store.load("chat-1") is None


def test_checkpoint_saves_when_descriptor_chmod_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delattr(session_store.os, "fchmod")
    memory = WorkingMemory(thread_id="windows-python-311")
    turn = memory.begin_turn("lưu được với Python Windows")
    memory.finish_turn(turn.sequence, "checkpoint đã tạo")

    path = SessionCheckpointStore(tmp_path / "sessions").save(memory)

    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8"))["thread_id"] == "windows-python-311"


def test_checkpoint_skips_directory_sync_on_windows(tmp_path: Path, monkeypatch) -> None:
    def directory_open(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("Windows must not open a directory for fsync")

    monkeypatch.setattr(session_store.os, "name", "nt")
    monkeypatch.setattr(session_store.os, "open", directory_open)

    session_store._sync_checkpoint_directory(tmp_path)


def test_checkpoint_rejects_non_private_file(tmp_path: Path) -> None:
    store = SessionCheckpointStore(tmp_path / "sessions")
    path = store._path("chat-1")
    path.write_text('{"version":1,"thread_id":"chat-1","turns":[],"generation":0,"revision":0,"summary":null}')
    path.chmod(0o644)
    with pytest.raises(ValueError, match="permissions"):
        store.load("chat-1")


def test_checkpoint_rejects_symlinked_parent_directory(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(real_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")

    with pytest.raises(ValueError, match="parent must not be a symlink"):
        SessionCheckpointStore(linked_parent / "sessions")

    assert not (real_parent / "sessions").exists()


def test_checkpoint_reads_legacy_working_payload(tmp_path: Path) -> None:
    memory = WorkingMemory(thread_id="legacy")
    turn = memory.begin_turn("cũ")
    memory.finish_turn(turn.sequence, "đã đọc")
    store = SessionCheckpointStore(tmp_path / "sessions")
    path = store._path("legacy")
    path.write_text(json.dumps(memory.to_dict()), encoding="utf-8")
    path.chmod(0o600)

    loaded = store.load("legacy")

    assert loaded is not None
    assert loaded.snapshot.turns == memory.snapshot.turns


def test_checkpoint_rejects_stale_equal_revision_state(tmp_path: Path) -> None:
    store = SessionCheckpointStore(tmp_path / "sessions")
    first = WorkingMemory(thread_id="same")
    turn = first.begin_turn("first")
    first.finish_turn(turn.sequence, "one")
    path = store.save(first)
    loaded, revision, digest = store.load_with_metadata("same")
    assert loaded is not None
    assert revision == first.snapshot.revision
    assert digest is not None

    divergent = WorkingMemory(thread_id="same")
    turn = divergent.begin_turn("other")
    divergent.finish_turn(turn.sequence, "two")
    divergent._revision = first.snapshot.revision  # noqa: SLF001 - adversarial fixture
    with pytest.raises(ValueError, match="advance revision"):
        store.save(divergent, expected_revision=revision, expected_digest=digest)
    assert path.exists()
