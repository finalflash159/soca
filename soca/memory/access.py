"""Explicit memory access policy passed to prompt assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MemoryArchiveMode = Literal["none", "semantic", "episodic", "both"]
# Compatibility alias for callers that only need the archive/not-archive boundary.
ArchiveMode = MemoryArchiveMode


@dataclass(frozen=True)
class MemoryAccessPlan:
    include_core: bool = True
    include_working: bool = True
    archive_mode: MemoryArchiveMode = "none"
    archive_query: str | None = None
    reason: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.include_core, bool) or not isinstance(self.include_working, bool):
            raise ValueError("memory access flags must be boolean")
        if self.archive_mode not in {"none", "semantic", "episodic", "both"}:
            raise ValueError("unknown memory archive mode")
        if self.archive_mode == "none" and self.archive_query is not None:
            raise ValueError("archive_query requires an archive mode")
        if self.archive_mode != "none" and not (self.archive_query or "").strip():
            raise ValueError("archive modes require a non-empty archive_query")
        if not self.reason.strip():
            raise ValueError("memory access reason must not be empty")


__all__ = ["ArchiveMode", "MemoryArchiveMode", "MemoryAccessPlan"]
