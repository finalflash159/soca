from __future__ import annotations

import json
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from soca.core.workflow import GoalCheckpointStore, GoalContract
from soca.memory import SessionCheckpointStore
from soca.memory.session_repository import (
    SessionConflictError,
    SessionMigrationError,
    SessionNotFoundError,
    SessionPermissionError,
    SessionRepository,
    SessionRepositoryError,
    SessionSchemaError,
)
from soca.memory.working import WorkingMemory


def _working_payload(session_id: str) -> dict[str, object]:
    memory = WorkingMemory(thread_id=session_id)
    turn = memory.begin_turn("Quyết định lưu phiên")
    memory.finish_turn(turn.sequence, "Đã lưu context")
    return memory.to_dict()


def test_repository_creates_lists_renames_and_permanently_deletes_sessions(
    tmp_path: Path,
) -> None:
    repository = SessionRepository(tmp_path / "sessions")
    first = repository.create_session(title="Đọc kế hoạch barge-in")
    second = repository.create_session(title="Kiểm tra desktop")

    page = repository.list_sessions(limit=1)

    assert page.sessions == (second,)
    assert page.next_cursor is not None
    assert repository.list_sessions(limit=1, cursor=page.next_cursor).sessions == (first,)

    renamed = repository.rename_session(
        first.session_id,
        title="Kế hoạch đã đổi tên",
        expected_revision=first.revision,
    )
    assert renamed.title == "Kế hoạch đã đổi tên"
    assert renamed.revision == first.revision + 1

    assert repository.delete_session(second.session_id, expected_revision=second.revision) is True
    with pytest.raises(SessionNotFoundError):
        repository.snapshot(second.session_id)


def test_repository_commits_turn_and_checkpoints_as_one_snapshot(tmp_path: Path) -> None:
    repository = SessionRepository(tmp_path / "sessions")
    session = repository.create_session(title="Phiên mới")
    pending = repository.begin_turn(
        session.session_id,
        user_text="Câu hỏi đầu tiên",
        surface="chat",
        working_checkpoint=_working_payload(session.session_id),
    )

    completed = repository.complete_turn(
        session.session_id,
        pending.turn_id,
        assistant_text="Câu trả lời cuối cùng",
        terminal_status="achieved",
        route="knowledge",
        citations=({"label": "K1", "path": "wiki/plan.md", "line_start": 3},),
        usage={"prompt_tokens": 12, "completion_tokens": 8},
        working_checkpoint=_working_payload(session.session_id),
        goal_checkpoint={"goal": None, "last_run": None},
    )
    snapshot = repository.snapshot(session.session_id)

    assert completed.status == "complete"
    assert completed.assistant_text == "Câu trả lời cuối cùng"
    assert snapshot.session.turn_count == 1
    assert snapshot.turns == (completed,)
    assert snapshot.working_checkpoint == _working_payload(session.session_id)
    assert snapshot.goal_checkpoint == {"goal": None, "last_run": None}


def test_repository_marks_pending_turn_interrupted_without_replaying_it(tmp_path: Path) -> None:
    repository = SessionRepository(tmp_path / "sessions")
    session = repository.create_session(title="Crash recovery")
    pending = repository.begin_turn(
        session.session_id,
        user_text="Đang xử lý",
        surface="voice",
        working_checkpoint=_working_payload(session.session_id),
    )

    recovered = repository.recover_interrupted_turns(session.session_id)
    snapshot = repository.snapshot(session.session_id)

    assert recovered == (pending.turn_id,)
    assert snapshot.turns[0].status == "interrupted"
    assert snapshot.turns[0].error_code == "process_terminated"
    assert snapshot.turns[0].assistant_text is None


def test_repository_rejects_stale_session_revision(tmp_path: Path) -> None:
    repository = SessionRepository(tmp_path / "sessions")
    session = repository.create_session(title="CAS")
    repository.rename_session(session.session_id, title="Mới", expected_revision=session.revision)

    with pytest.raises(SessionConflictError, match="revision changed"):
        repository.rename_session(
            session.session_id, title="Cũ", expected_revision=session.revision
        )


def test_repository_preferences_have_private_defaults_and_validate_active_session(
    tmp_path: Path,
) -> None:
    repository = SessionRepository(tmp_path / "sessions")
    session = repository.create_session(title="Tùy chọn")

    assert repository.get_preferences().auto_open_last is False
    stored = repository.set_preferences(
        auto_open_last=True,
        last_active_session_id=session.session_id,
    )

    assert stored.auto_open_last is True
    assert stored.last_active_session_id == session.session_id
    assert repository.get_preferences() == stored


def test_repository_database_is_private_and_uses_safe_pragmas(tmp_path: Path) -> None:
    repository = SessionRepository(tmp_path / "sessions")
    repository.create_session(title="Private")

    assert stat.S_IMODE(repository.database_path.stat().st_mode) == 0o600
    connection = repository._connect()  # noqa: SLF001 - assert repository connection contract
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert connection.execute("PRAGMA secure_delete").fetchone()[0] == 1
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        connection.close()


def test_repository_rejects_symlinked_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    try:
        linked_root.symlink_to(real_root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")

    with pytest.raises(SessionPermissionError, match="parent must not be a symlink"):
        SessionRepository(linked_root / "sessions")


def test_repository_rejects_unknown_database_schema_without_resetting_it(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    repository = SessionRepository(root)
    connection = sqlite3.connect(repository.database_path)
    try:
        connection.execute("PRAGMA user_version = 2")
    finally:
        connection.close()

    with pytest.raises(SessionSchemaError, match="unsupported session repository schema: 2"):
        SessionRepository(root)

    connection = sqlite3.connect(repository.database_path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
    finally:
        connection.close()


def test_repository_migrates_legacy_checkpoints_as_checkpoint_only(tmp_path: Path) -> None:
    legacy_root = tmp_path / "legacy"
    session_id = "legacy-thread"
    working = WorkingMemory(thread_id=session_id)
    turn = working.begin_turn("Nội dung context cũ")
    working.finish_turn(turn.sequence, "Câu trả lời cũ")
    SessionCheckpointStore(legacy_root).save(working)
    GoalCheckpointStore(legacy_root / "goals").save(
        session_id,
        goal=GoalContract(goal_id="goal-1", objective="Kiểm tra migration"),
        last_run=None,
    )

    repository = SessionRepository(tmp_path / "sessions")
    report = repository.migrate_legacy_checkpoints(legacy_root)

    assert report.imported == 1
    migrated = repository.list_sessions(limit=10).sessions
    assert len(migrated) == 1
    assert migrated[0].checkpoint_only is True
    snapshot = repository.snapshot(migrated[0].session_id)
    assert snapshot.turns == ()
    assert snapshot.working_checkpoint is not None
    assert snapshot.working_checkpoint["thread_id"] == session_id
    assert snapshot.goal_checkpoint is not None
    assert snapshot.goal_checkpoint["goal"]["goal_id"] == "goal-1"


def test_repository_migration_rejects_corrupt_legacy_checkpoint(tmp_path: Path) -> None:
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir(mode=0o700)
    corrupt = legacy_root / "broken.json"
    corrupt.write_text("not-json", encoding="utf-8")
    corrupt.chmod(0o600)

    repository = SessionRepository(tmp_path / "sessions")

    with pytest.raises(SessionMigrationError, match="cannot decode"):
        repository.migrate_legacy_checkpoints(legacy_root)

    assert repository.list_sessions(limit=10).sessions == ()


def test_repository_migration_is_idempotent_and_keeps_private_backup_manifest(
    tmp_path: Path,
) -> None:
    legacy_root = tmp_path / "legacy"
    memory = WorkingMemory(thread_id="legacy-thread")
    SessionCheckpointStore(legacy_root).save(memory)
    repository = SessionRepository(tmp_path / "sessions")

    first = repository.migrate_legacy_checkpoints(legacy_root)
    second = repository.migrate_legacy_checkpoints(legacy_root)

    assert first.imported == 1
    assert second.already_migrated is True
    assert stat.S_IMODE(first.backup_manifest.stat().st_mode) == 0o400
    assert stat.S_IMODE(first.backup_manifest.parent.stat().st_mode) == 0o500
    manifest = json.loads(first.backup_manifest.read_text(encoding="utf-8"))
    assert manifest["source_root"] == str(legacy_root)
    assert manifest["files"]


def test_repository_rolls_back_terminal_turn_when_checkpoint_write_fails(tmp_path: Path) -> None:
    repository = SessionRepository(tmp_path / "sessions")
    session = repository.create_session(title="Atomic terminal")
    pending = repository.begin_turn(
        session.session_id,
        user_text="Câu hỏi",
        surface="chat",
        working_checkpoint=_working_payload(session.session_id),
    )
    connection = sqlite3.connect(repository.database_path)
    try:
        connection.execute(
            """
            CREATE TRIGGER reject_goal_checkpoint
            BEFORE INSERT ON goal_checkpoints
            BEGIN SELECT RAISE(ABORT, 'goal checkpoint unavailable'); END
            """
        )
    finally:
        connection.close()

    with pytest.raises(SessionRepositoryError, match="cannot commit terminal session turn"):
        repository.complete_turn(
            session.session_id,
            pending.turn_id,
            assistant_text="Không được commit một nửa",
            terminal_status="achieved",
            route="knowledge",
            citations=(),
            usage=None,
            working_checkpoint=_working_payload(session.session_id),
            goal_checkpoint={"goal": None, "last_run": None},
        )

    snapshot = repository.snapshot(session.session_id)
    assert snapshot.turns[0].status == "pending"
    assert snapshot.turns[0].assistant_text is None


def test_repository_lease_rejects_another_process_until_released(tmp_path: Path) -> None:
    repository = SessionRepository(tmp_path / "sessions")
    script = """
from pathlib import Path
import sys
from soca.memory.session_repository import SessionConflictError, SessionRepository

repository = SessionRepository(Path(sys.argv[1]))
try:
    with repository.exclusive_lease():
        raise SystemExit(1)
except SessionConflictError:
    raise SystemExit(0)
"""

    with repository.exclusive_lease():
        blocked = subprocess.run(
            [sys.executable, "-c", script, str(repository.root)],
            cwd=Path.cwd(),
            check=False,
        )
    acquired = subprocess.run(
        [sys.executable, "-c", script, str(repository.root)],
        cwd=Path.cwd(),
        check=False,
    )

    assert blocked.returncode == 0
    assert acquired.returncode == 1


def test_repository_migration_requires_the_exclusive_lease(tmp_path: Path) -> None:
    repository = SessionRepository(tmp_path / "sessions")
    script = """
from pathlib import Path
import sys
from soca.memory.session_repository import SessionConflictError, SessionRepository

repository = SessionRepository(Path(sys.argv[1]))
try:
    repository.migrate_legacy_checkpoints(Path(sys.argv[2]))
except SessionConflictError:
    raise SystemExit(0)
raise SystemExit(1)
"""
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir(mode=0o700)

    with repository.exclusive_lease():
        blocked = subprocess.run(
            [sys.executable, "-c", script, str(repository.root), str(legacy_root)],
            cwd=Path.cwd(),
            check=False,
        )

    assert blocked.returncode == 0
