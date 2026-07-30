from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from soca.core.workflow import (
    ActiveGoalStore,
    GoalCheckpointConflictError,
    GoalCheckpointCorruptError,
    GoalCheckpointPermissionError,
    GoalCheckpointStore,
    GoalConstraint,
    GoalContract,
    GoalStatus,
    SourceKind,
    SuccessCriterion,
    WorkflowRunCheckpoint,
)


def _goal() -> GoalContract:
    return GoalContract(
        goal_id="goal-1",
        objective="Tìm ghi chú Bayes",
        success_criteria=(SuccessCriterion("knowledge_queried", source=SourceKind.KNOWLEDGE),),
        constraints=(GoalConstraint("scope", "vault"),),
        required_sources=(SourceKind.KNOWLEDGE,),
    )


def test_goal_checkpoint_round_trip_is_private_and_resumable(tmp_path: Path) -> None:
    store = GoalCheckpointStore(tmp_path / "goals")
    first = ActiveGoalStore(checkpoint_store=store, session_id="session-1")
    first.set(_goal())

    resumed = ActiveGoalStore(checkpoint_store=store, session_id="session-1")

    assert resumed.current == first.current
    assert resumed.last_run is None
    assert store.root.stat().st_mode & 0o077 == 0
    assert resumed.current is not None
    assert store._path("session-1").stat().st_mode & 0o077 == 0


def test_goal_checkpoint_rejects_corruption_without_silent_reset(tmp_path: Path) -> None:
    store = GoalCheckpointStore(tmp_path / "goals")
    store.save("session-1", goal=_goal(), last_run=None)
    target = store.path_for("session-1")
    target.write_text("not json", encoding="utf-8")

    with pytest.raises(GoalCheckpointCorruptError, match="cannot read goal checkpoint"):
        ActiveGoalStore(checkpoint_store=store, session_id="session-1")


def test_goal_checkpoint_rejects_symlink_root_and_detects_conflict(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(GoalCheckpointPermissionError):
        GoalCheckpointStore(linked_root)

    store = GoalCheckpointStore(real_root / "goals")
    store.save("session-1", goal=_goal(), last_run=None)
    loaded = store.load("session-1")
    store.save(
        "session-1",
        goal=None,
        last_run=None,
        expected_revision=loaded.revision,
        expected_digest=loaded.digest,
    )

    with pytest.raises(GoalCheckpointConflictError):
        store.save(
            "session-1",
            goal=_goal(),
            last_run=None,
            expected_revision=loaded.revision,
            expected_digest=loaded.digest,
        )


def test_goal_checkpoint_clear_persists_terminal_goal_state(tmp_path: Path) -> None:
    store = GoalCheckpointStore(tmp_path / "goals")
    active = ActiveGoalStore(checkpoint_store=store, session_id="session-1")
    active.set(_goal())
    active.clear()

    assert ActiveGoalStore(checkpoint_store=store, session_id="session-1").current is None


def test_goal_checkpoint_restores_last_run_identity_and_terminal_status(tmp_path: Path) -> None:
    store = GoalCheckpointStore(tmp_path / "goals")
    store.save(
        "session-1",
        goal=_goal(),
        last_run=WorkflowRunCheckpoint(
            run_id="run-1",
            goal_id="goal-1",
            terminal_status="waiting_for_user",
            updated_at="2026-07-30T00:00:00+00:00",
        ),
    )

    resumed = ActiveGoalStore(checkpoint_store=store, session_id="session-1")

    assert resumed.last_run is not None
    assert resumed.last_run.run_id == "run-1"
    assert resumed.last_run.terminal_status == "waiting_for_user"


def test_checkpoint_goal_serialization_preserves_status() -> None:
    goal = _goal()
    restored = GoalContract.from_checkpoint_dict(goal.to_checkpoint_dict())

    assert restored.status is GoalStatus.ACTIVE
    assert restored.required_sources == (SourceKind.KNOWLEDGE,)


def test_shared_goal_store_serializes_concurrent_goal_and_run_writes(tmp_path: Path) -> None:
    checkpoint_store = GoalCheckpointStore(tmp_path / "goals")
    active = ActiveGoalStore(checkpoint_store=checkpoint_store, session_id="session-1")
    goal = _goal()
    run = SimpleNamespace(
        state=SimpleNamespace(run_id="run-1", goal=goal),
        terminal=SimpleNamespace(status=SimpleNamespace(value="achieved")),
    )

    def write_goal() -> None:
        for _ in range(20):
            active.set(goal)

    def write_run() -> None:
        for _ in range(20):
            active.record_run(run)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(write_goal), pool.submit(write_run)]
        for future in futures:
            future.result()

    restored = ActiveGoalStore(checkpoint_store=checkpoint_store, session_id="session-1")
    assert restored.current == goal
    assert restored.last_run is not None
    assert restored.last_run.run_id == "run-1"
