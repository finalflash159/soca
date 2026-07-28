"""Private, versioned local checkpoints for opt-in resumable working memory."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from soca.memory.working import WorkingMemory


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
        payload = json.dumps(memory.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")
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
        return WorkingMemory.from_dict(json.loads(target.read_text(encoding="utf-8")))

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


__all__ = ["SessionCheckpointStore"]
