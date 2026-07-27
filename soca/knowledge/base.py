from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class KnowledgeDocument:
    id: str
    path: str
    title: str
    text: str
    tags: tuple[str, ...] = ()
    frontmatter: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgeHit:
    document: KnowledgeDocument
    score: float
    snippet: str


class KnowledgeSource(Protocol):
    def search(self, query: str, limit: int = 5) -> list[KnowledgeHit]:
        ...

    def read(self, path: str) -> KnowledgeDocument:
        ...
