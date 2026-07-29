"""Private, versioned local checkpoints for opt-in resumable working memory."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from soca.memory.working import WorkingMemory

CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointConflictError(ValueError):
    """Raised when a newer checkpoint would be overwritten."""


def default_session_checkpoint_home() -> Path:
    configured = os.environ.get("XDG_STATE_HOME", "").strip()
    base = Path(configured).expanduser() if configured else Path.home() / ".local" / "state"
    return base / "soca" / "sessions"


class SessionCheckpointStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError("session checkpoint root must be a real directory")

    def save(self, memory: WorkingMemory) -> Path:
        target = self._path(memory.thread_id)
        if target.exists() and target.is_symlink():
            raise ValueError("session checkpoint must not be a symlink")
        current = self._read_payload(target) if target.exists() else None
        current_revision = _payload_revision(current)
        if current_revision is not None and current_revision > memory.snapshot.revision:
            raise CheckpointConflictError("session checkpoint is newer than working memory")
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
        descriptor, temporary = tempfile.mkstemp(prefix=".working-", suffix=".json", dir=self.root)
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
        target = self._path(thread_id)
        if not target.exists():
            return None
        if target.is_symlink() or not target.is_file():
            raise ValueError("session checkpoint must be a real file")
        if target.stat().st_mode & 0o077:
            raise ValueError("session checkpoint permissions must be private")
        return WorkingMemory.from_dict(_working_payload(self._read_payload(target)))

    def delete(self, thread_id: str) -> bool:
        target = self._path(thread_id)
        if not target.exists():
            return False
        if target.is_symlink() or not target.is_file():
            raise ValueError("session checkpoint must be a real file")
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


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointConflictError",
    "SessionCheckpointStore",
    "default_session_checkpoint_home",
]
