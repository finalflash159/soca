"""Engine-owned lifecycle for RAM and locally resumable conversation sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from soca.memory import (
    PersistedTurn,
    SessionNotFoundError,
    SessionPage,
    SessionPreferences,
    SessionRecord,
    SessionRepository,
    SessionSnapshot,
)

SessionPersistenceMode = Literal["ram_only", "local_resumable"]
SessionSurface = Literal["chat", "voice"]


@dataclass(frozen=True)
class ActiveSession:
    session_id: str
    persistence: SessionPersistenceMode
    revision: int | None


@dataclass(frozen=True)
class SessionTurn:
    session_id: str
    turn_id: str
    sequence: int
    surface: SessionSurface


class SessionCoordinator:
    """The engine's single owner for durable session lifecycle operations.

    RAM-only sessions get IDs for protocol isolation but never create repository
    rows. A locally resumable coordinator delegates every persistent mutation to
    ``SessionRepository`` and keeps only the active identity in memory.
    """

    def __init__(
        self,
        *,
        persistence: SessionPersistenceMode,
        repository: SessionRepository | None = None,
        session_id: str | None = None,
    ) -> None:
        if persistence == "local_resumable" and repository is None:
            raise ValueError("local_resumable sessions require a repository")
        if persistence == "ram_only" and repository is not None:
            raise ValueError("ram_only sessions must not receive a repository")
        self.persistence = persistence
        self.repository = repository
        self._ram_sequence = 1
        self._active = ActiveSession(
            session_id=session_id or str(uuid4()),
            persistence=persistence,
            revision=None,
        )

    @property
    def active(self) -> ActiveSession:
        return self._active

    def initialize(self, *, title: str = "Cuộc trò chuyện mới") -> SessionSnapshot | None:
        """Create or recover the active persisted aggregate without replaying work."""
        if self.repository is None:
            return None
        try:
            snapshot = self.repository.snapshot(self._active.session_id)
        except SessionNotFoundError:
            record = self.repository.create_session(
                title=title,
                session_id=self._active.session_id,
            )
            self._active = ActiveSession(
                session_id=record.session_id,
                persistence="local_resumable",
                revision=record.revision,
            )
            return self.repository.snapshot(record.session_id)
        self.repository.recover_interrupted_turns(snapshot.session.session_id)
        recovered = self.repository.snapshot(snapshot.session.session_id)
        self._active = ActiveSession(
            session_id=recovered.session.session_id,
            persistence="local_resumable",
            revision=recovered.session.revision,
        )
        return recovered

    def create(self, *, title: str = "Cuộc trò chuyện mới") -> SessionSnapshot | None:
        if self.repository is None:
            self._active = ActiveSession(
                session_id=str(uuid4()), persistence="ram_only", revision=None
            )
            self._ram_sequence = 1
            return None
        record = self.repository.create_session(title=title)
        self._active = ActiveSession(
            session_id=record.session_id,
            persistence="local_resumable",
            revision=record.revision,
        )
        return self.repository.snapshot(record.session_id)

    def list(self, *, limit: int, cursor: str | None = None) -> SessionPage:
        if self.repository is None:
            return SessionPage(sessions=(), next_cursor=None)
        return self.repository.list_sessions(limit=limit, cursor=cursor)

    def preferences(self) -> SessionPreferences:
        if self.repository is None:
            return SessionPreferences(auto_open_last=False, last_active_session_id=None)
        return self.repository.get_preferences()

    def set_preferences(self, *, auto_open_last: bool) -> SessionPreferences:
        if self.repository is None:
            if auto_open_last:
                raise RuntimeError("auto-open requires locally saved sessions")
            return SessionPreferences(auto_open_last=False, last_active_session_id=None)
        return self.repository.set_preferences(
            auto_open_last=auto_open_last,
            last_active_session_id=self._active.session_id,
        )

    def open(self, session_id: str, *, busy: bool) -> SessionSnapshot:
        if self.repository is None:
            raise RuntimeError("saved sessions are disabled")
        if busy:
            raise SessionBusyError("cannot open a session while a turn or voice capture is active")
        self.repository.recover_interrupted_turns(session_id)
        snapshot = self.repository.snapshot(session_id)
        self._active = ActiveSession(
            session_id=snapshot.session.session_id,
            persistence="local_resumable",
            revision=snapshot.session.revision,
        )
        return snapshot

    def rename(self, session_id: str, *, title: str, expected_revision: int) -> SessionRecord:
        if self.repository is None:
            raise RuntimeError("saved sessions are disabled")
        record = self.repository.rename_session(
            session_id, title=title, expected_revision=expected_revision
        )
        if record.session_id == self._active.session_id:
            self._active = ActiveSession(
                session_id=record.session_id,
                persistence="local_resumable",
                revision=record.revision,
            )
        return record

    def delete(
        self, session_id: str, *, expected_revision: int, busy: bool
    ) -> SessionSnapshot | None:
        if self.repository is None:
            raise RuntimeError("saved sessions are disabled")
        if busy:
            raise SessionBusyError(
                "cannot delete a session while a turn or voice capture is active"
            )
        self.repository.delete_session(session_id, expected_revision=expected_revision)
        if session_id != self._active.session_id:
            return None
        return self.create()

    def begin_turn(
        self,
        *,
        user_text: str,
        surface: SessionSurface,
        working_checkpoint: dict[str, Any],
    ) -> SessionTurn:
        if self.repository is None:
            turn = SessionTurn(
                session_id=self._active.session_id,
                turn_id=str(uuid4()),
                sequence=self._ram_sequence,
                surface=surface,
            )
            self._ram_sequence += 1
            return turn
        persisted = self.repository.begin_turn(
            self._active.session_id,
            user_text=user_text,
            surface=surface,
            working_checkpoint=working_checkpoint,
        )
        self._refresh_active_revision()
        return _turn(persisted)

    def complete_turn(
        self,
        turn: SessionTurn,
        *,
        assistant_text: str | None,
        terminal_status: str | None,
        route: str | None,
        citations: tuple[dict[str, Any], ...],
        usage: dict[str, Any] | None,
        working_checkpoint: dict[str, Any],
        goal_checkpoint: dict[str, Any] | None,
        blocked: bool = False,
        repair_text: str | None = None,
        status: Literal["complete", "interrupted", "failed"] = "complete",
        error_code: str | None = None,
        error_detail: str | None = None,
    ) -> SessionTurn:
        if turn.session_id != self._active.session_id:
            raise SessionBusyError("cannot complete a turn from an inactive session")
        if self.repository is None:
            return turn
        persisted = self.repository.complete_turn(
            turn.session_id,
            turn.turn_id,
            assistant_text=assistant_text,
            terminal_status=terminal_status,
            route=route,
            citations=citations,
            usage=usage,
            working_checkpoint=working_checkpoint,
            goal_checkpoint=goal_checkpoint,
            blocked=blocked,
            repair_text=repair_text,
            status=status,
            error_code=error_code,
            error_detail=error_detail,
        )
        self._refresh_active_revision()
        return _turn(persisted)

    def snapshot(self, *, limit: int, before_sequence: int | None = None) -> SessionSnapshot:
        if self.repository is None:
            raise RuntimeError("saved sessions are disabled")
        return self.repository.snapshot(
            self._active.session_id,
            limit=limit,
            before_sequence=before_sequence,
        )

    def _refresh_active_revision(self) -> None:
        if self.repository is None:
            return
        snapshot = self.repository.snapshot(self._active.session_id, limit=1)
        self._active = ActiveSession(
            session_id=self._active.session_id,
            persistence="local_resumable",
            revision=snapshot.session.revision,
        )


class SessionBusyError(RuntimeError):
    """A lifecycle operation was requested while the current runtime is active."""


def _turn(value: PersistedTurn) -> SessionTurn:
    return SessionTurn(
        session_id=value.session_id,
        turn_id=value.turn_id,
        sequence=value.sequence,
        surface=value.surface,
    )


__all__ = ["ActiveSession", "SessionBusyError", "SessionCoordinator", "SessionTurn"]
