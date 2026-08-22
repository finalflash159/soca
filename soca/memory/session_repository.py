"""Private, transactional storage for opt-in resumable conversation sessions."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import sqlite3
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from soca.memory.working import WorkingMemory

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows must report the unavailable lease explicitly.
    fcntl = None

SESSION_SCHEMA_VERSION = 1
PRIVATE_DIRECTORY_MODE = stat.S_IRWXU
PRIVATE_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR
PRIVATE_READ_ONLY_DIRECTORY_MODE = stat.S_IRUSR | stat.S_IXUSR
PRIVATE_READ_ONLY_FILE_MODE = stat.S_IRUSR
_MIGRATION_SETTING = "legacy_json_migration_v1"
_PREFERENCES_SETTING = "session_preferences_v1"

SessionSurface = Literal["chat", "voice"]
SessionTurnStatus = Literal["pending", "complete", "interrupted", "failed"]
SessionState = Literal["idle", "running", "degraded", "corrupt"]


class SessionRepositoryError(RuntimeError):
    """Base error for durable session storage."""


class SessionConflictError(SessionRepositoryError):
    """A caller attempted to write a session that changed since it was read."""


class SessionNotFoundError(SessionRepositoryError):
    """The requested session no longer exists."""


class SessionSchemaError(SessionRepositoryError):
    """The database schema is unavailable or unsupported."""


class SessionPermissionError(SessionRepositoryError):
    """The session store path is not private or is redirected by a symlink."""


class SessionMigrationError(SessionRepositoryError):
    """Legacy checkpoint migration cannot safely be completed."""


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    title: str
    preview: str
    state: SessionState
    persistence: Literal["local_resumable"]
    created_at: str
    updated_at: str
    last_opened_at: str
    revision: int
    next_turn_sequence: int
    turn_count: int
    checkpoint_only: bool


@dataclass(frozen=True)
class PersistedTurn:
    session_id: str
    turn_id: str
    sequence: int
    surface: SessionSurface
    user_text: str
    assistant_text: str | None
    repair_text: str | None
    status: SessionTurnStatus
    terminal_status: str | None
    blocked: bool
    route: str | None
    citations: tuple[dict[str, Any], ...]
    usage: dict[str, Any] | None
    created_at: str
    completed_at: str | None
    error_code: str | None
    error_detail: str | None


@dataclass(frozen=True)
class SessionPage:
    sessions: tuple[SessionRecord, ...]
    next_cursor: str | None


@dataclass(frozen=True)
class SessionSnapshot:
    session: SessionRecord
    turns: tuple[PersistedTurn, ...]
    next_turn_cursor: int | None
    working_checkpoint: dict[str, Any] | None
    goal_checkpoint: dict[str, Any] | None


@dataclass(frozen=True)
class MigrationReport:
    imported: int
    already_migrated: bool
    backup_manifest: Path


@dataclass(frozen=True)
class SessionPreferences:
    auto_open_last: bool
    last_active_session_id: str | None


def default_session_repository_home() -> Path:
    configured = os.environ.get("XDG_STATE_HOME", "").strip()
    base = Path(configured).expanduser() if configured else Path.home() / ".local" / "state"
    return base / "soca" / "sessions"


class SessionRepository:
    """The sole durable owner of persisted transcript and runtime session state."""

    def __init__(self, root: str | Path) -> None:
        requested_root = Path(root).expanduser().absolute()
        _reject_symlink_ancestors(requested_root)
        root_path = requested_root.resolve()
        try:
            root_path.mkdir(mode=PRIVATE_DIRECTORY_MODE, parents=True, exist_ok=True)
        except OSError as exc:
            raise SessionPermissionError("cannot create session repository root") from exc
        _reject_symlink_ancestors(root_path)
        root_stat = root_path.lstat()
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            raise SessionPermissionError("session repository root must be a real directory")
        os.chmod(root_path, PRIVATE_DIRECTORY_MODE)
        self.root = root_path
        self.database_path = self.root / "sessions.sqlite3"
        self._validate_database_path()
        connection = self._connect()
        try:
            self._ensure_schema(connection)
            self._assert_integrity(connection, full=False)
        except sqlite3.Error as exc:
            raise SessionSchemaError("cannot initialize session repository") from exc
        finally:
            connection.close()
        self._ensure_private_file(self.database_path)

    def create_session(self, *, title: str, session_id: str | None = None) -> SessionRecord:
        identifier = str(uuid4()) if session_id is None else _validate_uuid(session_id)
        now = _now()
        normalized_title = _normalize_title(title)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    INSERT INTO sessions(
                        session_id, legacy_session_id, title, preview, state, persistence,
                        checkpoint_only, created_at, updated_at, last_opened_at, revision,
                        next_turn_sequence, turn_count
                    ) VALUES (?, NULL, ?, '', 'idle', 'local_resumable', 0, ?, ?, ?, 1, 1, 0)
                    """,
                    (identifier, normalized_title, now, now, now),
                )
                record = self._select_session(connection, identifier)
                connection.commit()
            except sqlite3.Error as exc:
                connection.rollback()
                raise SessionRepositoryError("cannot create session") from exc
            except Exception:
                connection.rollback()
                raise
        finally:
            connection.close()
        return record

    def list_sessions(self, *, limit: int = 50, cursor: str | None = None) -> SessionPage:
        bounded_limit = _validate_limit(limit)
        cursor_key = _decode_cursor(cursor) if cursor is not None else None
        connection = self._connect()
        try:
            if cursor_key is None:
                rows = connection.execute(
                    """
                    SELECT * FROM sessions
                    ORDER BY updated_at DESC, session_id DESC
                    LIMIT ?
                    """,
                    (bounded_limit + 1,),
                ).fetchall()
            else:
                updated_at, session_id = cursor_key
                rows = connection.execute(
                    """
                    SELECT * FROM sessions
                    WHERE updated_at < ? OR (updated_at = ? AND session_id < ?)
                    ORDER BY updated_at DESC, session_id DESC
                    LIMIT ?
                    """,
                    (updated_at, updated_at, session_id, bounded_limit + 1),
                ).fetchall()
        finally:
            connection.close()
        records = tuple(_record_from_row(row) for row in rows[:bounded_limit])
        next_cursor = None
        if len(rows) > bounded_limit and records:
            last = records[-1]
            next_cursor = _encode_cursor(last.updated_at, last.session_id)
        return SessionPage(sessions=records, next_cursor=next_cursor)

    def get_preferences(self) -> SessionPreferences:
        connection = self._connect()
        try:
            payload = self._read_setting(connection, _PREFERENCES_SETTING)
        finally:
            connection.close()
        if payload is None:
            return SessionPreferences(auto_open_last=False, last_active_session_id=None)
        auto_open_last = payload.get("auto_open_last", False)
        last_active_session_id = payload.get("last_active_session_id")
        if not isinstance(auto_open_last, bool):
            raise SessionSchemaError("session preference auto_open_last is invalid")
        if last_active_session_id is not None:
            if not isinstance(last_active_session_id, str):
                raise SessionSchemaError("session preference last_active_session_id is invalid")
            _validate_uuid(last_active_session_id)
        return SessionPreferences(
            auto_open_last=auto_open_last,
            last_active_session_id=last_active_session_id,
        )

    def set_preferences(
        self,
        *,
        auto_open_last: bool,
        last_active_session_id: str | None,
    ) -> SessionPreferences:
        if not isinstance(auto_open_last, bool):
            raise ValueError("auto_open_last must be a boolean")
        if last_active_session_id is not None:
            last_active_session_id = _validate_uuid(last_active_session_id)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if last_active_session_id is not None:
                    self._select_session(connection, last_active_session_id)
                self._write_setting(
                    connection,
                    _PREFERENCES_SETTING,
                    {
                        "auto_open_last": auto_open_last,
                        "last_active_session_id": last_active_session_id,
                    },
                    now=_now(),
                )
                connection.commit()
            except sqlite3.Error as exc:
                connection.rollback()
                raise SessionRepositoryError("cannot update session preferences") from exc
            except Exception:
                connection.rollback()
                raise
        finally:
            connection.close()
        return SessionPreferences(
            auto_open_last=auto_open_last,
            last_active_session_id=last_active_session_id,
        )

    def rename_session(
        self,
        session_id: str,
        *,
        title: str,
        expected_revision: int,
    ) -> SessionRecord:
        normalized_title = _normalize_title(title)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                current = self._select_session(connection, session_id)
                _check_revision(current, expected_revision)
                now = _now()
                connection.execute(
                    """
                    UPDATE sessions
                    SET title=?, updated_at=?, revision=revision + 1
                    WHERE session_id=?
                    """,
                    (normalized_title, now, session_id),
                )
                record = self._select_session(connection, session_id)
                connection.commit()
            except sqlite3.Error as exc:
                connection.rollback()
                raise SessionRepositoryError("cannot rename session") from exc
            except Exception:
                connection.rollback()
                raise
        finally:
            connection.close()
        return record

    def begin_turn(
        self,
        session_id: str,
        *,
        user_text: str,
        surface: SessionSurface,
        working_checkpoint: Mapping[str, Any],
    ) -> PersistedTurn:
        if surface not in {"chat", "voice"}:
            raise ValueError("session surface must be chat or voice")
        statement = _required_text(user_text, "user_text")
        checkpoint = _json_object(working_checkpoint, "working_checkpoint")
        turn_id = str(uuid4())
        now = _now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                session = self._select_session(connection, session_id)
                sequence = session.next_turn_sequence
                connection.execute(
                    """
                    INSERT INTO turns(
                        turn_id, session_id, sequence, surface, user_text, assistant_text,
                        repair_text, status, terminal_status, blocked, route, citations_json,
                        usage_json, created_at, completed_at, error_code, error_detail
                    ) VALUES (?, ?, ?, ?, ?, NULL, NULL, 'pending', NULL, 0, NULL, '[]',
                        NULL, ?, NULL, NULL, NULL)
                    """,
                    (turn_id, session_id, sequence, surface, statement, now),
                )
                self._write_checkpoint(
                    connection,
                    table="working_checkpoints",
                    session_id=session_id,
                    payload=checkpoint,
                    now=now,
                )
                connection.execute(
                    """
                    UPDATE sessions
                    SET state='running', updated_at=?, revision=revision + 1,
                        next_turn_sequence=next_turn_sequence + 1, turn_count=turn_count + 1,
                        preview=?
                    WHERE session_id=?
                    """,
                    (now, _preview(statement), session_id),
                )
                turn = self._select_turn(connection, session_id, turn_id)
                connection.commit()
            except sqlite3.Error as exc:
                connection.rollback()
                raise SessionRepositoryError("cannot begin session turn") from exc
            except Exception:
                connection.rollback()
                raise
        finally:
            connection.close()
        return turn

    def complete_turn(
        self,
        session_id: str,
        turn_id: str,
        *,
        assistant_text: str | None,
        terminal_status: str | None,
        route: str | None,
        citations: tuple[Mapping[str, Any], ...],
        usage: Mapping[str, Any] | None,
        working_checkpoint: Mapping[str, Any],
        goal_checkpoint: Mapping[str, Any] | None,
        blocked: bool = False,
        repair_text: str | None = None,
        status: SessionTurnStatus = "complete",
        error_code: str | None = None,
        error_detail: str | None = None,
    ) -> PersistedTurn:
        if status not in {"complete", "interrupted", "failed"}:
            raise ValueError("terminal turn status must be complete, interrupted or failed")
        if assistant_text is not None and not assistant_text.strip():
            raise ValueError("assistant_text must be non-empty when provided")
        checkpoint = _json_object(working_checkpoint, "working_checkpoint")
        goal = (
            _json_object(goal_checkpoint, "goal_checkpoint")
            if goal_checkpoint is not None
            else None
        )
        citation_payload = [_json_object(citation, "citation") for citation in citations]
        usage_payload = _json_object(usage, "usage") if usage is not None else None
        now = _now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._select_session(connection, session_id)
                pending = self._select_turn(connection, session_id, turn_id)
                if pending.status != "pending":
                    raise SessionConflictError("turn is no longer pending")
                connection.execute(
                    """
                    UPDATE turns
                    SET assistant_text=?, repair_text=?, status=?, terminal_status=?, blocked=?, route=?,
                        citations_json=?, usage_json=?, completed_at=?, error_code=?, error_detail=?
                    WHERE session_id=? AND turn_id=?
                    """,
                    (
                        assistant_text,
                        repair_text,
                        status,
                        terminal_status,
                        int(blocked),
                        route,
                        _canonical_json(citation_payload),
                        _canonical_json(usage_payload) if usage_payload is not None else None,
                        now,
                        error_code,
                        error_detail,
                        session_id,
                        turn_id,
                    ),
                )
                self._write_checkpoint(
                    connection,
                    table="working_checkpoints",
                    session_id=session_id,
                    payload=checkpoint,
                    now=now,
                )
                if goal is not None:
                    self._write_checkpoint(
                        connection,
                        table="goal_checkpoints",
                        session_id=session_id,
                        payload=goal,
                        now=now,
                    )
                preview = _preview(assistant_text or repair_text or pending.user_text)
                connection.execute(
                    """
                    UPDATE sessions
                    SET state='idle', updated_at=?, revision=revision + 1, preview=?
                    WHERE session_id=?
                    """,
                    (now, preview, session_id),
                )
                turn = self._select_turn(connection, session_id, turn_id)
                connection.commit()
            except sqlite3.Error as exc:
                connection.rollback()
                raise SessionRepositoryError("cannot commit terminal session turn") from exc
            except Exception:
                connection.rollback()
                raise
        finally:
            connection.close()
        return turn

    def recover_interrupted_turns(self, session_id: str) -> tuple[str, ...]:
        now = _now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._select_session(connection, session_id)
                rows = connection.execute(
                    """
                    SELECT turn_id FROM turns
                    WHERE session_id=? AND status='pending'
                    ORDER BY sequence
                    """,
                    (session_id,),
                ).fetchall()
                turn_ids = tuple(str(row["turn_id"]) for row in rows)
                if turn_ids:
                    connection.execute(
                        """
                        UPDATE turns
                        SET status='interrupted', completed_at=?, error_code='process_terminated',
                            error_detail='The engine stopped before this turn reached a terminal outcome.'
                        WHERE session_id=? AND status='pending'
                        """,
                        (now, session_id),
                    )
                    connection.execute(
                        """
                        UPDATE sessions
                        SET state='idle', updated_at=?, revision=revision + 1
                        WHERE session_id=?
                        """,
                        (now, session_id),
                    )
                connection.commit()
            except sqlite3.Error as exc:
                connection.rollback()
                raise SessionRepositoryError("cannot recover interrupted session turns") from exc
            except Exception:
                connection.rollback()
                raise
        finally:
            connection.close()
        return turn_ids

    def snapshot(
        self,
        session_id: str,
        *,
        limit: int = 50,
        before_sequence: int | None = None,
    ) -> SessionSnapshot:
        bounded_limit = _validate_limit(limit)
        connection = self._connect()
        try:
            session = self._select_session(connection, session_id)
            params: tuple[object, ...]
            if before_sequence is None:
                params = (session_id, bounded_limit + 1)
                rows = connection.execute(
                    """
                    SELECT * FROM turns WHERE session_id=?
                    ORDER BY sequence DESC LIMIT ?
                    """,
                    params,
                ).fetchall()
            else:
                if before_sequence <= 0:
                    raise ValueError("before_sequence must be positive")
                rows = connection.execute(
                    """
                    SELECT * FROM turns WHERE session_id=? AND sequence < ?
                    ORDER BY sequence DESC LIMIT ?
                    """,
                    (session_id, before_sequence, bounded_limit + 1),
                ).fetchall()
            selected = rows[:bounded_limit]
            turns = tuple(reversed(tuple(_turn_from_row(row) for row in selected)))
            next_turn_cursor = turns[0].sequence if len(rows) > bounded_limit and turns else None
            working = self._read_checkpoint(connection, "working_checkpoints", session_id)
            goal = self._read_checkpoint(connection, "goal_checkpoints", session_id)
        finally:
            connection.close()
        return SessionSnapshot(
            session=session,
            turns=turns,
            next_turn_cursor=next_turn_cursor,
            working_checkpoint=working,
            goal_checkpoint=goal,
        )

    def delete_session(self, session_id: str, *, expected_revision: int) -> bool:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                session = self._select_session(connection, session_id)
                _check_revision(session, expected_revision)
                deleted = connection.execute(
                    "DELETE FROM sessions WHERE session_id=?", (session_id,)
                ).rowcount
                connection.commit()
            except sqlite3.Error as exc:
                connection.rollback()
                raise SessionRepositoryError("cannot delete session") from exc
            except Exception:
                connection.rollback()
                raise
        finally:
            connection.close()
        return deleted == 1

    def migrate_legacy_checkpoints(self, legacy_root: str | Path) -> MigrationReport:
        with self.exclusive_lease():
            return self._migrate_legacy_checkpoints(legacy_root)

    def _migrate_legacy_checkpoints(self, legacy_root: str | Path) -> MigrationReport:
        source_root = Path(legacy_root).expanduser().absolute()
        _reject_symlink_ancestors(source_root)
        connection = self._connect()
        try:
            marker = self._read_setting(connection, _MIGRATION_SETTING)
        finally:
            connection.close()
        if marker is not None:
            manifest = Path(str(marker.get("backup_manifest", "")))
            if not manifest.is_file():
                raise SessionMigrationError(
                    "legacy migration marker has no readable backup manifest"
                )
            return MigrationReport(imported=0, already_migrated=True, backup_manifest=manifest)

        backup_manifest = self._backup_legacy_source(source_root)
        legacy = _read_legacy_checkpoints(source_root)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if self._read_setting(connection, _MIGRATION_SETTING) is not None:
                    connection.rollback()
                    return MigrationReport(
                        imported=0,
                        already_migrated=True,
                        backup_manifest=backup_manifest,
                    )
                now = _now()
                for legacy_id, payload in legacy.items():
                    session_id = str(uuid4())
                    title = _legacy_title(payload)
                    connection.execute(
                        """
                        INSERT INTO sessions(
                            session_id, legacy_session_id, title, preview, state, persistence,
                            checkpoint_only, created_at, updated_at, last_opened_at, revision,
                            next_turn_sequence, turn_count
                        ) VALUES (?, ?, ?, ?, 'idle', 'local_resumable', 1, ?, ?, ?, 1, 1, 0)
                        """,
                        (
                            session_id,
                            legacy_id,
                            title,
                            _legacy_preview(payload),
                            now,
                            now,
                            now,
                        ),
                    )
                    working = payload.get("working")
                    if working is not None:
                        self._write_checkpoint(
                            connection,
                            table="working_checkpoints",
                            session_id=session_id,
                            payload=working,
                            now=now,
                        )
                    goal = payload.get("goal")
                    if goal is not None:
                        self._write_checkpoint(
                            connection,
                            table="goal_checkpoints",
                            session_id=session_id,
                            payload=goal,
                            now=now,
                        )
                self._write_setting(
                    connection,
                    _MIGRATION_SETTING,
                    {"backup_manifest": str(backup_manifest), "completed_at": now},
                    now=now,
                )
                self._assert_integrity(connection, full=True)
                self._assert_migration_round_trip(connection, legacy)
                connection.commit()
            except sqlite3.Error as exc:
                connection.rollback()
                raise SessionMigrationError("cannot import legacy session checkpoints") from exc
            except Exception:
                connection.rollback()
                raise
        finally:
            connection.close()
        return MigrationReport(
            imported=len(legacy),
            already_migrated=False,
            backup_manifest=backup_manifest,
        )

    def verify(self) -> tuple[str, ...]:
        connection = self._connect()
        try:
            problems = self._integrity_problems(connection, full=True)
        finally:
            connection.close()
        if stat.S_IMODE(self.database_path.stat().st_mode) != PRIVATE_FILE_MODE:
            problems.append("database_permissions")
        return tuple(problems)

    @contextmanager
    def exclusive_lease(self) -> Iterator[None]:
        """Prevent a second engine process from mutating the same session store."""
        if fcntl is None:
            raise SessionRepositoryError(
                "exclusive session leases are unavailable on this platform"
            )
        lock_path = self.root / ".session-repository.lock"
        if lock_path.exists() and lock_path.is_symlink():
            raise SessionPermissionError("session repository lease must not be a symlink")
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, PRIVATE_FILE_MODE)
            os.chmod(lock_path, PRIVATE_FILE_MODE)
        except OSError as exc:
            raise SessionRepositoryError("cannot open session repository lease") from exc
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise SessionConflictError("session repository lease is already owned") from exc
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _connect(self) -> sqlite3.Connection:
        self._validate_database_path()
        try:
            connection = sqlite3.connect(self.database_path, timeout=5.0, isolation_level=None)
        except sqlite3.Error as exc:
            raise SessionRepositoryError("cannot open session repository database") from exc
        connection.row_factory = sqlite3.Row
        try:
            _set_pragma(connection, "foreign_keys", "ON", expected=1)
            _set_pragma(connection, "busy_timeout", "5000", expected=5000)
            _set_pragma(connection, "journal_mode", "DELETE", expected="delete")
            _set_pragma(connection, "synchronous", "FULL", expected=2)
            _set_pragma(connection, "trusted_schema", "OFF", expected=0)
            _set_pragma(connection, "secure_delete", "ON", expected=1)
        except sqlite3.Error as exc:
            connection.close()
            raise SessionSchemaError("cannot configure session repository database") from exc
        except Exception:
            connection.close()
            raise
        self._ensure_private_file(self.database_path)
        return connection

    @staticmethod
    def _integrity_problems(connection: sqlite3.Connection, *, full: bool) -> list[str]:
        pragma = "integrity_check" if full else "quick_check"
        problems = [
            str(row[0]) for row in connection.execute(f"PRAGMA {pragma}") if str(row[0]) != "ok"
        ]
        problems.extend(
            f"foreign_key:{row[0]}:{row[1]}"
            for row in connection.execute("PRAGMA foreign_key_check")
        )
        return problems

    def _assert_integrity(self, connection: sqlite3.Connection, *, full: bool) -> None:
        problems = self._integrity_problems(connection, full=full)
        if problems:
            check = "integrity" if full else "quick"
            raise SessionSchemaError(f"session repository {check} check failed: {problems[0]}")

    @staticmethod
    def _assert_migration_round_trip(
        connection: sqlite3.Connection,
        legacy: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ) -> None:
        for legacy_id, payload in legacy.items():
            row = connection.execute(
                "SELECT session_id FROM sessions WHERE legacy_session_id=?", (legacy_id,)
            ).fetchone()
            if row is None:
                raise SessionMigrationError("legacy migration session is missing after import")
            session_id = str(row["session_id"])
            checkpoints: tuple[
                tuple[
                    Literal["working", "goal"],
                    Literal["working_checkpoints", "goal_checkpoints"],
                ],
                ...,
            ] = (
                ("working", "working_checkpoints"),
                ("goal", "goal_checkpoints"),
            )
            for checkpoint_name, table in checkpoints:
                expected = payload.get(checkpoint_name)
                actual = SessionRepository._read_checkpoint(connection, table, session_id)
                if actual != expected:
                    raise SessionMigrationError(
                        f"legacy {checkpoint_name} checkpoint did not round-trip safely"
                    )

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        current = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if current > SESSION_SCHEMA_VERSION:
            raise SessionSchemaError(f"unsupported session repository schema: {current}")
        if current == SESSION_SCHEMA_VERSION:
            return
        if current != 0:
            raise SessionSchemaError(f"cannot migrate session repository schema: {current}")
        now = _now()
        connection.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                legacy_session_id TEXT UNIQUE,
                title TEXT NOT NULL,
                preview TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('idle', 'running', 'degraded', 'corrupt')),
                persistence TEXT NOT NULL CHECK(persistence = 'local_resumable'),
                checkpoint_only INTEGER NOT NULL CHECK(checkpoint_only IN (0, 1)),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_opened_at TEXT NOT NULL,
                revision INTEGER NOT NULL CHECK(revision >= 1),
                next_turn_sequence INTEGER NOT NULL CHECK(next_turn_sequence >= 1),
                turn_count INTEGER NOT NULL CHECK(turn_count >= 0)
            );
            CREATE INDEX sessions_updated_order ON sessions(updated_at DESC, session_id DESC);
            CREATE TABLE turns (
                turn_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                sequence INTEGER NOT NULL CHECK(sequence >= 1),
                surface TEXT NOT NULL CHECK(surface IN ('chat', 'voice')),
                user_text TEXT NOT NULL,
                assistant_text TEXT,
                repair_text TEXT,
                status TEXT NOT NULL CHECK(status IN ('pending', 'complete', 'interrupted', 'failed')),
                terminal_status TEXT,
                blocked INTEGER NOT NULL CHECK(blocked IN (0, 1)),
                route TEXT,
                citations_json TEXT NOT NULL,
                usage_json TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                error_code TEXT,
                error_detail TEXT,
                UNIQUE(session_id, sequence),
                FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );
            CREATE INDEX turns_session_sequence ON turns(session_id, sequence DESC);
            CREATE TABLE turn_steps (
                session_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                node TEXT NOT NULL,
                status TEXT NOT NULL,
                public_message TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY(turn_id, sequence),
                FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
                FOREIGN KEY(turn_id) REFERENCES turns(turn_id) ON DELETE CASCADE
            );
            CREATE TABLE working_checkpoints (
                session_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                digest TEXT NOT NULL,
                revision INTEGER NOT NULL CHECK(revision >= 1),
                updated_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );
            CREATE TABLE goal_checkpoints (
                session_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                digest TEXT NOT NULL,
                revision INTEGER NOT NULL CHECK(revision >= 1),
                updated_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );
            CREATE TABLE session_settings (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );
            PRAGMA user_version = 1;
            """
        )
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
            (SESSION_SCHEMA_VERSION, now),
        )

    def _select_session(self, connection: sqlite3.Connection, session_id: str) -> SessionRecord:
        row = connection.execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if row is None:
            raise SessionNotFoundError(f"session does not exist: {session_id}")
        return _record_from_row(row)

    @staticmethod
    def _select_turn(
        connection: sqlite3.Connection, session_id: str, turn_id: str
    ) -> PersistedTurn:
        row = connection.execute(
            "SELECT * FROM turns WHERE session_id=? AND turn_id=?", (session_id, turn_id)
        ).fetchone()
        if row is None:
            raise SessionNotFoundError(f"turn does not exist: {turn_id}")
        return _turn_from_row(row)

    @staticmethod
    def _write_checkpoint(
        connection: sqlite3.Connection,
        *,
        table: Literal["working_checkpoints", "goal_checkpoints"],
        session_id: str,
        payload: Mapping[str, Any],
        now: str,
    ) -> None:
        encoded = _canonical_json(dict(payload))
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        connection.execute(
            f"""
            INSERT INTO {table}(session_id, payload_json, digest, revision, updated_at)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                payload_json=excluded.payload_json,
                digest=excluded.digest,
                revision={table}.revision + 1,
                updated_at=excluded.updated_at
            """,
            (session_id, encoded, digest, now),
        )

    @staticmethod
    def _read_checkpoint(
        connection: sqlite3.Connection,
        table: Literal["working_checkpoints", "goal_checkpoints"],
        session_id: str,
    ) -> dict[str, Any] | None:
        row = connection.execute(
            f"SELECT payload_json FROM {table} WHERE session_id=?", (session_id,)
        ).fetchone()
        if row is None:
            return None
        try:
            value = json.loads(str(row["payload_json"]))
        except json.JSONDecodeError as exc:
            raise SessionSchemaError(f"invalid {table} payload") from exc
        return _json_object(value, table)

    @staticmethod
    def _read_setting(connection: sqlite3.Connection, key: str) -> dict[str, Any] | None:
        row = connection.execute(
            "SELECT value_json FROM session_settings WHERE key=?", (key,)
        ).fetchone()
        if row is None:
            return None
        try:
            value = json.loads(str(row["value_json"]))
        except json.JSONDecodeError as exc:
            raise SessionSchemaError(f"invalid session setting: {key}") from exc
        return _json_object(value, key)

    @staticmethod
    def _write_setting(
        connection: sqlite3.Connection, key: str, value: Mapping[str, Any], *, now: str
    ) -> None:
        connection.execute(
            """
            INSERT INTO session_settings(key, value_json, updated_at) VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
            """,
            (key, _canonical_json(dict(value)), now),
        )

    def _backup_legacy_source(self, source_root: Path) -> Path:
        backup_root = self.root / "legacy-backups" / uuid4().hex
        backup_root.mkdir(mode=PRIVATE_DIRECTORY_MODE, parents=True, exist_ok=False)
        os.chmod(backup_root, PRIVATE_DIRECTORY_MODE)
        source_files = (
            *_legacy_files(source_root, include_goals=False),
            *_legacy_files(source_root),
        )
        manifest_files: list[dict[str, str | int]] = []
        backup_directories = {backup_root}
        for source in source_files:
            relative = source.relative_to(source_root)
            target = backup_root / relative
            target.parent.mkdir(mode=PRIVATE_DIRECTORY_MODE, parents=True, exist_ok=True)
            os.chmod(target.parent, PRIVATE_DIRECTORY_MODE)
            backup_directories.add(target.parent)
            shutil.copyfile(source, target)
            os.chmod(target, PRIVATE_READ_ONLY_FILE_MODE)
            manifest_files.append(
                {
                    "path": str(relative),
                    "sha256": _file_digest(source),
                    "bytes": source.stat().st_size,
                    "mode": stat.S_IMODE(source.stat().st_mode),
                }
            )
        manifest = backup_root / "manifest.json"
        manifest.write_text(
            _canonical_json(
                {
                    "source_root": str(source_root),
                    "created_at": _now(),
                    "files": manifest_files,
                    "database_schema_version": SESSION_SCHEMA_VERSION,
                    "sqlite_version": sqlite3.sqlite_version,
                }
            ),
            encoding="utf-8",
        )
        os.chmod(manifest, PRIVATE_READ_ONLY_FILE_MODE)
        for directory in sorted(backup_directories, key=lambda item: len(item.parts), reverse=True):
            os.chmod(directory, PRIVATE_READ_ONLY_DIRECTORY_MODE)
        return manifest

    def _validate_database_path(self) -> None:
        if self.database_path.exists():
            metadata = self.database_path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise SessionPermissionError("session database must be a real file")
            if metadata.st_mode & 0o077:
                raise SessionPermissionError("session database permissions must be private")

    @staticmethod
    def _ensure_private_file(path: Path) -> None:
        try:
            os.chmod(path, PRIVATE_FILE_MODE)
        except FileNotFoundError:
            return


def _read_legacy_checkpoints(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    if not root.exists():
        return {}
    working_by_id: dict[str, dict[str, Any]] = {}
    for path in _legacy_files(root, include_goals=False):
        payload = _read_private_json(path, "working checkpoint")
        working_payload = _legacy_working_payload(payload)
        try:
            memory = WorkingMemory.from_dict(working_payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise SessionMigrationError(f"cannot decode working checkpoint: {path.name}") from exc
        working_by_id[memory.snapshot.thread_id] = memory.to_dict()

    goals_root = root / "goals"
    goal_by_id: dict[str, dict[str, Any]] = {}
    if goals_root.exists():
        for path in _legacy_files(root, include_goals=True):
            payload = _read_private_json(path, "goal checkpoint")
            session_id = payload.get("session_id") if isinstance(payload, dict) else None
            if not isinstance(session_id, str) or not session_id.strip():
                raise SessionMigrationError(f"cannot decode goal checkpoint: {path.name}")
            if payload.get("schema_version") != 1:
                raise SessionMigrationError(f"cannot decode goal checkpoint: {path.name}")
            goal = payload.get("goal")
            last_run = payload.get("last_run")
            if goal is not None and not isinstance(goal, dict):
                raise SessionMigrationError(f"cannot decode goal checkpoint: {path.name}")
            if last_run is not None and not isinstance(last_run, dict):
                raise SessionMigrationError(f"cannot decode goal checkpoint: {path.name}")
            goal_by_id[session_id] = {
                "goal": goal,
                "last_run": last_run,
            }

    result: dict[str, dict[str, dict[str, Any]]] = {}
    for session_id in sorted(set(working_by_id) | set(goal_by_id)):
        item: dict[str, dict[str, Any]] = {}
        if session_id in working_by_id:
            item["working"] = working_by_id[session_id]
        if session_id in goal_by_id:
            item["goal"] = goal_by_id[session_id]
        result[session_id] = item
    return result


def _legacy_files(root: Path, *, include_goals: bool = True) -> tuple[Path, ...]:
    if not root.exists():
        return ()
    _validate_private_directory(root, "legacy checkpoint root")
    directory = root / "goals" if include_goals else root
    if not directory.exists():
        return ()
    _validate_private_directory(directory, "legacy goal checkpoint root")
    files = tuple(sorted(directory.glob("*.json")))
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise SessionMigrationError("legacy checkpoint must be a real file")
        if path.stat().st_mode & 0o077:
            raise SessionMigrationError("legacy checkpoint permissions must be private")
    return files


def _read_private_json(path: Path, kind: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SessionMigrationError(f"cannot decode {kind}: {path.name}") from exc


def _legacy_working_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SessionMigrationError("cannot decode working checkpoint")
    wrapped = payload.get("working")
    if payload.get("schema_version") == 1 and isinstance(wrapped, dict):
        if payload.get("persistence") != "local_resumable":
            raise SessionMigrationError("unsupported working checkpoint persistence")
        return wrapped
    return payload


def _legacy_title(payload: Mapping[str, Mapping[str, Any]]) -> str:
    working = payload.get("working")
    if working is not None:
        turns = working.get("turns")
        if isinstance(turns, list):
            for turn in turns:
                if isinstance(turn, dict) and isinstance(turn.get("user_text"), str):
                    return _normalize_title(str(turn["user_text"]))
    goal = payload.get("goal")
    if goal is not None:
        goal_payload = goal.get("goal")
        if isinstance(goal_payload, dict) and isinstance(goal_payload.get("objective"), str):
            return _normalize_title(str(goal_payload["objective"]))
    return "Phiên đã khôi phục"


def _legacy_preview(payload: Mapping[str, Mapping[str, Any]]) -> str:
    working = payload.get("working")
    if working is None:
        return "Chỉ khôi phục context; không có transcript đầy đủ."
    turns = working.get("turns")
    if not isinstance(turns, list) or not turns:
        return "Chỉ khôi phục context; không có transcript đầy đủ."
    latest = turns[-1]
    if isinstance(latest, dict):
        text = latest.get("assistant_text") or latest.get("user_text")
        if isinstance(text, str) and text.strip():
            return _preview(text)
    return "Chỉ khôi phục context; không có transcript đầy đủ."


def _record_from_row(row: sqlite3.Row) -> SessionRecord:
    return SessionRecord(
        session_id=str(row["session_id"]),
        title=str(row["title"]),
        preview=str(row["preview"]),
        state=str(row["state"]),  # type: ignore[arg-type]
        persistence="local_resumable",
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        last_opened_at=str(row["last_opened_at"]),
        revision=int(row["revision"]),
        next_turn_sequence=int(row["next_turn_sequence"]),
        turn_count=int(row["turn_count"]),
        checkpoint_only=bool(row["checkpoint_only"]),
    )


def _turn_from_row(row: sqlite3.Row) -> PersistedTurn:
    citations = json.loads(str(row["citations_json"]))
    usage = json.loads(str(row["usage_json"])) if row["usage_json"] is not None else None
    if not isinstance(citations, list) or not all(isinstance(item, dict) for item in citations):
        raise SessionSchemaError("turn citation payload is invalid")
    if usage is not None and not isinstance(usage, dict):
        raise SessionSchemaError("turn usage payload is invalid")
    return PersistedTurn(
        session_id=str(row["session_id"]),
        turn_id=str(row["turn_id"]),
        sequence=int(row["sequence"]),
        surface=str(row["surface"]),  # type: ignore[arg-type]
        user_text=str(row["user_text"]),
        assistant_text=str(row["assistant_text"]) if row["assistant_text"] is not None else None,
        repair_text=str(row["repair_text"]) if row["repair_text"] is not None else None,
        status=str(row["status"]),  # type: ignore[arg-type]
        terminal_status=(
            str(row["terminal_status"]) if row["terminal_status"] is not None else None
        ),
        blocked=bool(row["blocked"]),
        route=str(row["route"]) if row["route"] is not None else None,
        citations=tuple(dict(item) for item in citations),
        usage=dict(usage) if usage is not None else None,
        created_at=str(row["created_at"]),
        completed_at=str(row["completed_at"]) if row["completed_at"] is not None else None,
        error_code=str(row["error_code"]) if row["error_code"] is not None else None,
        error_detail=str(row["error_detail"]) if row["error_detail"] is not None else None,
    )


def _check_revision(session: SessionRecord, expected_revision: int) -> None:
    if session.revision != expected_revision:
        raise SessionConflictError("session revision changed")


def _set_pragma(
    connection: sqlite3.Connection, name: str, value: str, *, expected: str | int
) -> None:
    connection.execute(f"PRAGMA {name} = {value}")
    row = connection.execute(f"PRAGMA {name}").fetchone()
    if row is None or row[0] != expected:
        raise SessionSchemaError(f"cannot apply SQLite pragma {name}")


def _normalize_title(value: str) -> str:
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise ValueError("session title must not be empty")
    return normalized[:48]


def _preview(value: str) -> str:
    return " ".join(value.strip().split())[:160]


def _required_text(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _validate_uuid(value: str) -> str:
    try:
        return str(UUID(value))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("session_id must be a UUID") from exc


def _validate_limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
        raise ValueError("limit must be an integer between one and 100")
    return value


def _json_object(value: Mapping[str, Any] | Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    result = dict(value)
    try:
        json.dumps(result, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON serializable") from exc
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _encode_cursor(updated_at: str, session_id: str) -> str:
    raw = _canonical_json({"updated_at": updated_at, "session_id": session_id}).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str) -> tuple[str, str]:
    try:
        padding = "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(cursor + padding).decode("utf-8"))
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid session cursor") from exc
    if not isinstance(payload, dict):
        raise ValueError("invalid session cursor")
    updated_at = payload.get("updated_at")
    session_id = payload.get("session_id")
    if not isinstance(updated_at, str) or not isinstance(session_id, str):
        raise ValueError("invalid session cursor")
    return updated_at, session_id


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_private_directory(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise SessionMigrationError(f"{label} must be a real directory")
    if path.stat().st_mode & 0o077:
        raise SessionMigrationError(f"{label} permissions must be private")


def _reject_symlink_ancestors(path: Path) -> None:
    for ancestor in (path, *path.parents):
        try:
            metadata = ancestor.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISLNK(metadata.st_mode):
            continue
        if ancestor == Path("/var") and ancestor.resolve() == Path("/private/var"):
            continue
        raise SessionPermissionError("session repository root or parent must not be a symlink")


__all__ = [
    "MigrationReport",
    "PersistedTurn",
    "SESSION_SCHEMA_VERSION",
    "SessionConflictError",
    "SessionMigrationError",
    "SessionNotFoundError",
    "SessionPage",
    "SessionPermissionError",
    "SessionPreferences",
    "SessionRecord",
    "SessionRepository",
    "SessionRepositoryError",
    "SessionSchemaError",
    "SessionSnapshot",
    "default_session_repository_home",
]
