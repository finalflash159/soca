from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import yaml

ALLOWED_KEYS = frozenset({"created_at", "updated_at", "importance"})


@dataclass(frozen=True)
class MemoryMetadata:
    created_at: datetime | None = None
    updated_at: datetime | None = None
    importance: int | None = None


def parse_memory_frontmatter(text: str, *, max_bytes: int = 8_192) -> MemoryMetadata:
    encoded = text.encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError("memory frontmatter is too large")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return MemoryMetadata()
    try:
        end = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration as exc:
        raise ValueError("memory frontmatter is unterminated") from exc
    raw = "\n".join(lines[1:end])
    for token in yaml.scan(raw):
        if token.__class__.__name__ in {"AnchorToken", "AliasToken", "TagToken"}:
            raise ValueError("memory frontmatter may not contain YAML anchors or tags")
    loaded = yaml.safe_load(raw) or {}
    if not isinstance(loaded, dict) or any(key not in ALLOWED_KEYS for key in loaded):
        raise ValueError("memory frontmatter has unsupported fields")
    if any(isinstance(value, (dict, list, tuple, set)) for value in loaded.values()):
        raise ValueError("memory frontmatter values must be scalar")
    importance = loaded.get("importance")
    if importance is not None and (
        isinstance(importance, bool) or not isinstance(importance, int) or not 1 <= importance <= 10
    ):
        raise ValueError("memory importance must be an integer between 1 and 10")
    created_at = _parse_timestamp(loaded.get("created_at"))
    updated_at = _parse_timestamp(loaded.get("updated_at"))
    if created_at is not None and updated_at is not None and updated_at < created_at:
        raise ValueError("memory timestamps are out of order")
    return MemoryMetadata(created_at=created_at, updated_at=updated_at, importance=importance)


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("memory timestamp must be ISO-8601") from exc
    else:
        raise ValueError("memory timestamp must be a string")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("memory timestamp must include a timezone")
    return parsed.astimezone(UTC)


__all__ = ["MemoryMetadata", "parse_memory_frontmatter"]
