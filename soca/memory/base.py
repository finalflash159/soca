from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

MemoryRole = Literal["user", "assistant"]


@dataclass(frozen=True)
class MemoryTurn:
    role: MemoryRole
    text: str


class LongTermMemorySource(Protocol):
    def read_profile(self) -> str:
        ...


class SessionMemorySource(Protocol):
    def append(self, role: MemoryRole, text: str) -> None:
        ...

    def render(self) -> str:
        ...

    def clear(self) -> None:
        ...
