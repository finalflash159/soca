from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

EPISODE_SCHEMA_VERSION = 1
MAX_SUMMARY_CHARS = 8_000
MAX_ITEMS = 32


@dataclass(frozen=True)
class MemoryEpisode:
    id: str
    created_at: datetime
    ended_at: datetime
    summary: str
    retained_facts: tuple[str, ...] = ()
    unresolved_items: tuple[str, ...] = ()
    schema_version: int = EPISODE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        try:
            uuid.UUID(self.id)
        except (ValueError, AttributeError) as exc:
            raise ValueError("episode id must be a UUID") from exc
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != EPISODE_SCHEMA_VERSION
        ):
            raise ValueError("unsupported episode schema version")
        if not self.summary.strip() or len(self.summary) > MAX_SUMMARY_CHARS:
            raise ValueError("episode summary is invalid")
        if len(self.retained_facts) > MAX_ITEMS or len(self.unresolved_items) > MAX_ITEMS:
            raise ValueError("episode item count is too large")
        if any(not item.strip() or len(item) > 1_000 for item in (*self.retained_facts, *self.unresolved_items)):
            raise ValueError("episode item is invalid")
        created = _utc(self.created_at)
        ended = _utc(self.ended_at)
        if ended < created:
            raise ValueError("episode timestamps are out of order")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "ended_at", ended)
        object.__setattr__(self, "retained_facts", tuple(self.retained_facts))
        object.__setattr__(self, "unresolved_items", tuple(self.unresolved_items))


class EpisodeStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        self._ensure_directory()
        self._manifest = self.root / "manifest.json"

    def persist(self, episode: MemoryEpisode) -> Path:
        self._ensure_directory()
        path = self._episode_path(episode.id)
        if path.exists() or path.is_symlink():
            existing = self._load_path(path)
            if existing != episode:
                raise ValueError("episode id already exists with different content")
            return path
        self._atomic_write(path, _encode_episode(episode), mode=0o600)
        self._write_manifest()
        return path

    def get(self, episode_id: str) -> MemoryEpisode | None:
        return self._load_path(self._episode_path(episode_id), missing_ok=True)

    def load_all(self) -> tuple[MemoryEpisode, ...]:
        self._ensure_directory()
        episodes: list[MemoryEpisode] = []
        for path in sorted(self.root.glob("*.json")):
            if path.name == "manifest.json":
                continue
            if path.is_symlink():
                raise ValueError("episode store contains a symlink")
            episode = self._load_path(path)
            if episode is None:
                raise ValueError(f"episode file is missing: {path}")
            episodes.append(episode)
        return tuple(episodes)

    def delete(self, episode_id: str) -> bool:
        path = self._episode_path(episode_id)
        if not path.exists():
            return False
        if path.is_symlink():
            raise ValueError("episode path may not be a symlink")
        path.unlink()
        self._write_manifest()
        return True

    def _episode_path(self, episode_id: str) -> Path:
        try:
            parsed = uuid.UUID(episode_id)
        except (ValueError, AttributeError) as exc:
            raise ValueError("episode id must be a UUID") from exc
        return self.root / f"{parsed}.json"

    def _load_path(self, path: Path, *, missing_ok: bool = False) -> MemoryEpisode | None:
        if not path.exists():
            if missing_ok:
                return None
            raise FileNotFoundError(path)
        if path.is_symlink():
            raise ValueError("episode path may not be a symlink")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return _decode_episode(payload)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError("episode file is invalid") from exc

    def _ensure_directory(self) -> None:
        if self.root.exists() and self.root.is_symlink():
            raise ValueError("episode directory may not be a symlink")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise ValueError("episode directory may not be a symlink")
        self.root.chmod(0o700)

    def _write_manifest(self) -> None:
        ids = [episode.id for episode in self.load_all()]
        payload = {"schema_version": EPISODE_SCHEMA_VERSION, "ids": sorted(ids)}
        self._atomic_write(self._manifest, _canonical_json(payload), mode=0o600)

    @staticmethod
    def _atomic_write(path: Path, data: bytes, *, mode: int) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            os.fchmod(descriptor, mode)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("episode timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _encode_episode(episode: MemoryEpisode) -> bytes:
    payload = asdict(episode)
    payload["created_at"] = episode.created_at.isoformat()
    payload["ended_at"] = episode.ended_at.isoformat()
    return _canonical_json(payload)


def _decode_episode(payload: object) -> MemoryEpisode:
    if not isinstance(payload, dict):
        raise ValueError("episode root must be an object")
    required = {"id", "created_at", "ended_at", "summary", "retained_facts", "unresolved_items", "schema_version"}
    if set(payload) != required:
        raise ValueError("episode schema is invalid")
    return MemoryEpisode(
        id=payload["id"],
        created_at=datetime.fromisoformat(payload["created_at"]),
        ended_at=datetime.fromisoformat(payload["ended_at"]),
        summary=payload["summary"],
        retained_facts=tuple(payload["retained_facts"]),
        unresolved_items=tuple(payload["unresolved_items"]),
        schema_version=payload["schema_version"],
    )


def _canonical_json(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


__all__ = ["EPISODE_SCHEMA_VERSION", "EpisodeStore", "MemoryEpisode"]
