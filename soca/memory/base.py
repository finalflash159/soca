from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

MemoryRole = Literal["user", "assistant"]


@dataclass(frozen=True)
class MemoryTurn:
    role: MemoryRole
    text: str


@dataclass(frozen=True)
class MemoryProfileResult:
    text: str
    hits: tuple[object, ...] = ()
    mode: str = "blob"
    degraded_reason: str = ""


class LongTermMemorySource(Protocol):
    def read_profile(self) -> str:
        ...


@runtime_checkable
class QueryAwareLongTermMemorySource(Protocol):
    def retrieve_profile(self, query: str) -> MemoryProfileResult:
        ...


class SessionMemorySource(Protocol):
    def append(self, role: MemoryRole, text: str) -> None:
        ...

    def render(self) -> str:
        ...

    def clear(self) -> None:
        ...
