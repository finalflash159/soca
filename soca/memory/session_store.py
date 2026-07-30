"""Private, versioned local checkpoints for opt-in resumable working memory."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from soca.memory.working import WorkingMemory, WorkingMemoryPolicy

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback is covered by API behavior.
    fcntl = None

CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointConflictError(ValueError):
    """Raised when a newer checkpoint would be overwritten."""


def default_session_checkpoint_home() -> Path:
    configured = os.environ.get("XDG_STATE_HOME", "").strip()
    base = Path(configured).expanduser() if configured else Path.home() / ".local" / "state"
    return base / "soca" / "sessions"


class SessionCheckpointStore:
    def __init__(self, root: str | Path) -> None:
        root_path = Path(root).expanduser().absolute()
        _reject_symlink_ancestors(root_path)
        root_path.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root_path.is_symlink() or not root_path.is_dir():
            raise ValueError("session checkpoint root must be a real directory")
        os.chmod(root_path, 0o700)
        self.root = root_path

    def save(
        self,
        memory: WorkingMemory,
        *,
        expected_revision: int | None = None,
        expected_digest: str | None = None,
    ) -> Path:
        target = self._path(memory.thread_id)
        with self._exclusive_lock():
            if target.exists() and target.is_symlink():
                raise ValueError("session checkpoint must not be a symlink")
            current = self._read_payload(target) if target.exists() else None
            _check_expected_state(
                current,
                expected_revision=expected_revision,
                expected_digest=expected_digest,
            )
            if (
                current is not None
                and _payload_revision(current) == memory.snapshot.revision
                and _payload_digest(current) != _payload_digest(memory.to_dict())
            ):
                raise CheckpointConflictError(
                    "session checkpoint update must advance revision"
                )
            payload = json.dumps(
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "thread_id": memory.thread_id,
                    "revision": memory.snapshot.revision,
                    "persistence": "local_resumable",
                    "working": memory.to_dict(),
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
            descriptor, temporary = tempfile.mkstemp(
                prefix=".working-", suffix=".json", dir=self.root
            )
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
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

    def load(self, thread_id: str) -> WorkingMemory | None:
        memory, _, _ = self.load_with_metadata(thread_id)
        return memory

    def load_with_metadata(
        self,
        thread_id: str,
        *,
        policy: WorkingMemoryPolicy | None = None,
    ) -> tuple[WorkingMemory | None, int | None, str | None]:
        target = self._path(thread_id)
        with self._exclusive_lock():
            if not target.exists():
                return None, None, None
            if target.is_symlink() or not target.is_file():
                raise ValueError("session checkpoint must be a real file")
            if target.stat().st_mode & 0o077:
                raise ValueError("session checkpoint permissions must be private")
            payload = self._read_payload(target)
            return (
                WorkingMemory.from_dict(_working_payload(payload), policy=policy),
                _payload_revision(payload),
                _payload_digest(payload),
            )

    def delete(
        self,
        thread_id: str,
        *,
        expected_revision: int | None = None,
        expected_digest: str | None = None,
    ) -> bool:
        target = self._path(thread_id)
        with self._exclusive_lock():
            if not target.exists():
                return False
            if target.is_symlink() or not target.is_file():
                raise ValueError("session checkpoint must be a real file")
            if expected_revision is not None or expected_digest is not None:
                _check_expected_state(
                    self._read_payload(target),
                    expected_revision=expected_revision,
                    expected_digest=expected_digest,
                )
            target.unlink()
            return True

    def _path(self, thread_id: str) -> Path:
        if not thread_id.strip():
            raise ValueError("thread_id must not be empty")
        digest = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    @staticmethod
    def _read_payload(target: Path) -> object:
        return json.loads(target.read_text(encoding="utf-8"))

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        lock_path = self.root / ".session.lock"
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.chmod(lock_path, 0o600)
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _working_payload(payload: object) -> object:
    if isinstance(payload, dict) and payload.get("schema_version") == CHECKPOINT_SCHEMA_VERSION:
        if payload.get("persistence") != "local_resumable":
            raise ValueError("unsupported checkpoint persistence mode")
        working = payload.get("working")
        if not isinstance(working, dict):
            raise ValueError("checkpoint working payload must be an object")
        return working
    # Version-1 working-memory checkpoints predate the session wrapper. They
    # remain readable, but every new write uses the wrapped schema above.
    return payload


def _payload_revision(payload: object) -> int | None:
    if isinstance(payload, dict) and payload.get("schema_version") == CHECKPOINT_SCHEMA_VERSION:
        value = payload.get("revision")
    elif isinstance(payload, dict):
        value = payload.get("revision")
    else:
        return None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _payload_digest(payload: object) -> str | None:
    try:
        working = _working_payload(payload)
    except ValueError:
        return None
    if not isinstance(working, dict):
        return None
    canonical = json.dumps(working, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _check_expected_state(
    current: object,
    *,
    expected_revision: int | None,
    expected_digest: str | None,
) -> None:
    if expected_revision is None and expected_digest is None:
        if current is not None:
            raise CheckpointConflictError("session checkpoint already exists")
        return
    if (
        _payload_revision(current) != expected_revision
        or _payload_digest(current) != expected_digest
    ):
        raise CheckpointConflictError("session checkpoint changed since it was read")


def _reject_symlink_ancestors(path: Path) -> None:
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError("session checkpoint root or parent must not be a symlink")


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointConflictError",
    "SessionCheckpointStore",
    "default_session_checkpoint_home",
]
