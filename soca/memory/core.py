"""Read-only approved core-memory store."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from soca.core.text_budget import truncate


@dataclass(frozen=True)
class CoreMemoryItem:
    id: str
    value: str
    approved_at: str
    sensitivity: str
    updated_at: str
    provenance: str


class CoreMemoryStore:
    """Load only explicitly approved items; never promote archive text."""

    def __init__(self, vault: str | Path, *, max_chars: int = 900) -> None:
        self.vault = Path(vault).expanduser().resolve()
        self.path = self.vault / "memory" / "core.json"
        if max_chars <= 0:
            raise ValueError("core memory max_chars must be positive")
        self.max_chars = max_chars

    def read_core(self) -> str:
        items = self.items()
        if not items:
            return ""
        rendered = "\n".join(f"- [{item.id}] {item.value}" for item in items)
        return truncate(rendered, self.max_chars)

    def items(self) -> tuple[CoreMemoryItem, ...]:
        if not self.path.exists():
            return ()
        if self.path.is_symlink() or not self.path.is_file():
            raise ValueError("core memory must be a regular file")
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("unsupported core memory schema")
        raw_items = payload.get("items")
        if not isinstance(raw_items, list):
            raise ValueError("core memory items must be a list")
        result: list[CoreMemoryItem] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                raise ValueError("core memory item must be an object")
            values = {
                key: str(raw.get(key, "")).strip()
                for key in (
                    "id",
                    "value",
                    "approved_at",
                    "sensitivity",
                    "updated_at",
                    "provenance",
                )
            }
            if any(not value for value in values.values()):
                raise ValueError("core memory item has missing metadata")
            result.append(CoreMemoryItem(**values))
        return tuple(result)


__all__ = ["CoreMemoryItem", "CoreMemoryStore"]
