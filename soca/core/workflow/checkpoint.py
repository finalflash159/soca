from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from .contracts import GoalContract

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows uses the same atomic file contract.
    fcntl = None

CHECKPOINT_SCHEMA_VERSION = 1


class GoalCheckpointError(RuntimeError):
    """Base error for an unreadable or conflicting goal checkpoint."""


class GoalCheckpointCorruptError(GoalCheckpointError):
    """Checkpoint bytes or fields cannot be decoded as the schema."""


class GoalCheckpointSchemaError(GoalCheckpointError):
    """Checkpoint schema version is not supported."""


class GoalCheckpointPermissionError(GoalCheckpointError):
    """Checkpoint path is not private or is redirected by a symlink."""


class GoalCheckpointConflictError(GoalCheckpointError):
    """A concurrent writer changed the checkpoint since it was loaded."""


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
    revision: int = 0
    digest: str | None = None


class GoalCheckpointStore:
    def __init__(self, root: str | Path) -> None:
        root_path = Path(root).expanduser().absolute()
        _reject_symlink_ancestors(root_path)
        try:
            root_path.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as exc:
            raise GoalCheckpointError("cannot create goal checkpoint root") from exc
        if root_path.is_symlink() or not root_path.is_dir():
            raise GoalCheckpointPermissionError(
                "goal checkpoint root must be a real directory"
            )
        os.chmod(root_path, 0o700)
        self.root = root_path

    def load(self, session_id: str) -> GoalCheckpoint:
        target = self._path(session_id)
        with self._exclusive_lock():
            if not target.exists():
                return GoalCheckpoint(goal=None, last_run=None)
            self._validate_target(target)
            try:
                payload: Any = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise GoalCheckpointCorruptError("cannot read goal checkpoint") from exc
            checkpoint = _decode(payload, session_id=session_id)
            return GoalCheckpoint(
                goal=checkpoint.goal,
                last_run=checkpoint.last_run,
                revision=checkpoint.revision,
                digest=_payload_digest(payload),
            )

    def save(
        self,
        session_id: str,
        *,
        goal: GoalContract | None,
        last_run: WorkflowRunCheckpoint | None,
        expected_revision: int | None = None,
        expected_digest: str | None = None,
    ) -> GoalCheckpoint:
        target = self._path(session_id)
        with self._exclusive_lock():
            current: Any | None = None
            if target.exists():
                self._validate_target(target)
                try:
                    current = json.loads(target.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise GoalCheckpointCorruptError("cannot read goal checkpoint") from exc
                _decode(current, session_id=session_id)
            _check_expected_state(
                current,
                expected_revision=expected_revision,
                expected_digest=expected_digest,
            )
            revision = _revision(current) + 1 if current is not None else 1
            payload = {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "session_id": session_id,
                "revision": revision,
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
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            descriptor, temporary = tempfile.mkstemp(
                prefix=".goal-", suffix=".json", dir=self.root
            )
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
        return GoalCheckpoint(
            goal=goal,
            last_run=last_run,
            revision=revision,
            digest=_payload_digest(payload),
        )

    def delete(self, session_id: str) -> bool:
        target = self._path(session_id)
        with self._exclusive_lock():
            if not target.exists():
                return False
            self._validate_target(target)
            target.unlink()
            return True

    def _path(self, session_id: str) -> Path:
        if not session_id.strip():
            raise GoalCheckpointError("session_id must not be empty")
        digest = sha256(session_id.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    def path_for(self, session_id: str) -> Path:
        return self._path(session_id)

    @staticmethod
    def _validate_target(target: Path) -> None:
        if target.is_symlink() or not target.is_file():
            raise GoalCheckpointPermissionError("goal checkpoint must be a real file")
        if target.stat().st_mode & 0o077:
            raise GoalCheckpointPermissionError("goal checkpoint permissions must be private")

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        lock_path = self.root / ".goal.lock"
        if lock_path.is_symlink():
            raise GoalCheckpointPermissionError("goal checkpoint lock must not be a symlink")
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            os.chmod(lock_path, 0o600)
        except OSError as exc:
            raise GoalCheckpointError("cannot open goal checkpoint lock") from exc
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _decode(payload: Any, *, session_id: str) -> GoalCheckpoint:
    if not isinstance(payload, dict):
        raise GoalCheckpointCorruptError("goal checkpoint must be an object")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise GoalCheckpointSchemaError("unsupported goal checkpoint schema")
    if payload.get("session_id") != session_id:
        raise GoalCheckpointCorruptError("goal checkpoint session id mismatch")
    try:
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
        return GoalCheckpoint(
            goal=goal,
            last_run=last_run,
            revision=_revision(payload),
        )
    except GoalCheckpointError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise GoalCheckpointCorruptError("invalid goal checkpoint") from exc


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"goal checkpoint {key} must be non-empty text")
    return value


def _revision(payload: Any) -> int:
    value = payload.get("revision", 0) if isinstance(payload, dict) else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GoalCheckpointCorruptError("goal checkpoint revision must be non-negative")
    return value


def _payload_digest(payload: Any) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha256(canonical).hexdigest()


def _check_expected_state(
    current: Any,
    *,
    expected_revision: int | None,
    expected_digest: str | None,
) -> None:
    if expected_revision is None and expected_digest is None:
        if current is not None:
            raise GoalCheckpointConflictError("goal checkpoint already exists")
        return
    if current is None or _revision(current) != expected_revision:
        raise GoalCheckpointConflictError("goal checkpoint revision changed")
    if expected_digest is not None and _payload_digest(current) != expected_digest:
        raise GoalCheckpointConflictError("goal checkpoint content changed")


def _reject_symlink_ancestors(path: Path) -> None:
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise GoalCheckpointPermissionError(
                "goal checkpoint root or parent must not be a symlink"
            )


def now_checkpoint_time() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "GoalCheckpoint",
    "GoalCheckpointConflictError",
    "GoalCheckpointCorruptError",
    "GoalCheckpointError",
    "GoalCheckpointPermissionError",
    "GoalCheckpointSchemaError",
    "GoalCheckpointStore",
    "WorkflowRunCheckpoint",
    "now_checkpoint_time",
]
