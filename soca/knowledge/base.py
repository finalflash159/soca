from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol


@dataclass(frozen=True)
class KnowledgeDocument:
    id: str
    path: str
    title: str
    text: str
    tags: tuple[str, ...] = ()
    frontmatter: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) for value in (self.id, self.path, self.title, self.text)):
            raise ValueError("document fields must be strings")
        if not self.id or not self.path:
            raise ValueError("document id and path must not be empty")
        if not isinstance(self.tags, (tuple, list)) or any(
            not isinstance(tag, str) for tag in self.tags
        ):
            raise ValueError("tags must contain strings")
        object.__setattr__(self, "tags", tuple(self.tags))
        if not isinstance(self.frontmatter, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.frontmatter.items()
        ):
            raise ValueError("frontmatter must map strings to strings")
        object.__setattr__(
            self,
            "frontmatter",
            MappingProxyType(dict(self.frontmatter)),
        )


@dataclass(frozen=True)
class KnowledgeHit:
    document: KnowledgeDocument
    score: float
    snippet: str
    line_start: int | None = None
    line_end: int | None = None
    retrieval_backend: str = "unknown"
    sparse_score: float | None = None
    dense_score: float | None = None
    fusion_score: float | None = None

    def __post_init__(self) -> None:
        if (self.line_start is None) != (self.line_end is None):
            raise ValueError("line_start and line_end must be provided together")
        if self.line_start is not None and (
            self.line_start < 1 or self.line_end is None or self.line_end < self.line_start
        ):
            raise ValueError("knowledge hit line range is invalid")
        for name, value in (
            ("score", self.score),
            ("sparse_score", self.sparse_score),
            ("dense_score", self.dense_score),
            ("fusion_score", self.fusion_score),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be a finite number or null")
        if not isinstance(self.retrieval_backend, str) or not self.retrieval_backend.strip():
            raise ValueError("retrieval_backend must be a non-empty string")


class KnowledgeSource(Protocol):
    def search(self, query: str, limit: int = 5) -> list[KnowledgeHit]: ...

    def read(self, path: str) -> KnowledgeDocument: ...
