from pathlib import Path

from soca.memory import SessionCheckpointStore, SessionMemory


def test_local_resumable_session_saves_resumes_and_clear_deletes(tmp_path: Path) -> None:
    store = SessionCheckpointStore(tmp_path / "sessions")
    first = SessionMemory(
        thread_id="thread-1",
        persistence="local_resumable",
        checkpoint_store=store,
        summary_enabled=False,
    )
    first.append("user", "giữ lượt này")
    first.append("assistant", "đã giữ")
    first.close()

    resumed = SessionMemory(
        thread_id="thread-1",
        persistence="local_resumable",
        checkpoint_store=store,
        resume=True,
        summary_enabled=False,
    )
    assert "giữ lượt này" in resumed.render()
    resumed.clear()
    assert store.load("thread-1") is None


def test_ram_only_session_does_not_write_checkpoint(tmp_path: Path) -> None:
    store = SessionCheckpointStore(tmp_path / "sessions")
    memory = SessionMemory(
        thread_id="ram",
        persistence="ram_only",
        checkpoint_store=store,
        summary_enabled=False,
    )
    memory.append("user", "không lưu đĩa")
    memory.append("assistant", "đúng")

    assert store.load("ram") is None


def test_local_session_exposes_private_checkpoint_path(tmp_path: Path) -> None:
    store = SessionCheckpointStore(tmp_path / "sessions")
    memory = SessionMemory(
        thread_id="path",
        persistence="local_resumable",
        checkpoint_store=store,
        summary_enabled=False,
    )

    assert memory.checkpoint_path is not None
    assert memory.checkpoint_path.parent == store.root
    assert memory.stats().persistence == "local_resumable"
    assert memory.stats().checkpoint_enabled is True
