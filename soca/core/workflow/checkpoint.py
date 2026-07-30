from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from .contracts import GoalContract

CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class WorkflowRunCheckpoint:
    run_id: str
    goal_id: str
    terminal_status: str
    updated_at: str


@dataclass(frozen=True)
class GoalCheckpoint:
    goal: GoalContract | None
    last_run: WorkflowRunCheckpoint | None


class GoalCheckpointStore:
    """Private, atomic checkpoints for resumable controlled-workflow goals."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError("goal checkpoint root must be a real directory")

    def load(self, session_id: str) -> GoalCheckpoint:
        target = self._path(session_id)
        if not target.exists():
            return GoalCheckpoint(goal=None, last_run=None)
        if target.is_symlink() or not target.is_file():
            raise ValueError("goal checkpoint must be a real file")
        if target.stat().st_mode & 0o077:
            raise ValueError("goal checkpoint permissions must be private")
        return _decode(target.read_text(encoding="utf-8"))

    def save(
        self,
        session_id: str,
        *,
        goal: GoalContract | None,
        last_run: WorkflowRunCheckpoint | None,
    ) -> Path:
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "session_id": session_id,
            "goal": goal.to_checkpoint_dict() if goal is not None else None,
            "last_run": (
                {
                    "run_id": last_run.run_id,
                    "goal_id": last_run.goal_id,
                    "terminal_status": last_run.terminal_status,
                    "updated_at": last_run.updated_at,
                }
                if last_run is not None
                else None
            ),
        }
        target = self._path(session_id)
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        descriptor, temporary = tempfile.mkstemp(prefix=".goal-", suffix=".json", dir=self.root)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            directory_fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            os.chmod(target, 0o600)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return target

    def delete(self, session_id: str) -> bool:
        target = self._path(session_id)
        if not target.exists():
            return False
        if target.is_symlink() or not target.is_file():
            raise ValueError("goal checkpoint must be a real file")
        target.unlink()
        return True

    def _path(self, session_id: str) -> Path:
        if not session_id.strip():
            raise ValueError("session_id must not be empty")
        digest = sha256(session_id.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"


def _decode(raw: str) -> GoalCheckpoint:
    try:
        payload: Any = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("goal checkpoint must be an object")
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported goal checkpoint schema")
        goal_data = payload.get("goal")
        goal = GoalContract.from_checkpoint_dict(goal_data) if goal_data is not None else None
        run_data = payload.get("last_run")
        if run_data is None:
            last_run = None
        elif isinstance(run_data, dict):
            last_run = WorkflowRunCheckpoint(
                run_id=_required_text(run_data, "run_id"),
                goal_id=_required_text(run_data, "goal_id"),
                terminal_status=_required_text(run_data, "terminal_status"),
                updated_at=_required_text(run_data, "updated_at"),
            )
        else:
            raise ValueError("goal checkpoint last_run must be an object")
        return GoalCheckpoint(goal=goal, last_run=last_run)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid goal checkpoint") from exc


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"goal checkpoint {key} must be non-empty text")
    return value


def now_checkpoint_time() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "GoalCheckpoint",
    "GoalCheckpointStore",
    "WorkflowRunCheckpoint",
    "now_checkpoint_time",
]
