from __future__ import annotations

import pytest

from soca.app.session_coordinator import SessionBusyError, SessionCoordinator
from soca.memory import SessionRepository


def test_ram_only_sessions_receive_isolated_ids_without_creating_files(tmp_path) -> None:
    coordinator = SessionCoordinator(persistence="ram_only")
    original = coordinator.active.session_id

    assert coordinator.initialize() is None
    assert coordinator.create() is None
    assert coordinator.active.session_id != original
    assert list(tmp_path.iterdir()) == []


def test_saved_session_lifecycle_is_owned_by_the_repository(tmp_path) -> None:
    coordinator = SessionCoordinator(
        persistence="local_resumable",
        repository=SessionRepository(tmp_path / "sessions"),
    )
    created = coordinator.initialize()

    assert created is not None
    assert coordinator.list(limit=10).sessions == (created.session,)

    pending = coordinator.begin_turn(
        user_text="Câu hỏi cần lưu",
        surface="chat",
        working_checkpoint={"thread_id": created.session.session_id, "turns": []},
    )
    coordinator.complete_turn(
        pending,
        assistant_text="Câu trả lời đã lưu",
        terminal_status="achieved",
        route="knowledge",
        citations=(),
        usage=None,
        working_checkpoint={"thread_id": created.session.session_id, "turns": []},
        goal_checkpoint=None,
    )
    reopened = coordinator.open(created.session.session_id, busy=False)

    assert reopened.turns[0].user_text == "Câu hỏi cần lưu"
    assert reopened.turns[0].assistant_text == "Câu trả lời đã lưu"


def test_active_saved_session_delete_creates_a_fresh_valid_active_session(tmp_path) -> None:
    coordinator = SessionCoordinator(
        persistence="local_resumable",
        repository=SessionRepository(tmp_path / "sessions"),
    )
    current = coordinator.initialize()
    assert current is not None

    replacement = coordinator.delete(
        current.session.session_id,
        expected_revision=current.session.revision,
        busy=False,
    )

    assert replacement is not None
    assert replacement.session.session_id != current.session.session_id
    assert coordinator.active.session_id == replacement.session.session_id


def test_session_switch_and_delete_are_rejected_while_runtime_is_busy(tmp_path) -> None:
    coordinator = SessionCoordinator(
        persistence="local_resumable",
        repository=SessionRepository(tmp_path / "sessions"),
    )
    current = coordinator.initialize()
    assert current is not None

    with pytest.raises(SessionBusyError):
        coordinator.open(current.session.session_id, busy=True)
    with pytest.raises(SessionBusyError):
        coordinator.delete(
            current.session.session_id,
            expected_revision=current.session.revision,
            busy=True,
        )
