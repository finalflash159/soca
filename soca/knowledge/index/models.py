from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from soca.knowledge.base import KnowledgeDocument

INDEX_VERSION = 1


def _require_relative_markdown_path(path: str) -> None:
    if not path or path.startswith("/") or "\\" in path:
        raise ValueError("path must be a non-empty POSIX relative path")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("path must not contain empty, dot, or parent segments")
    if not path.lower().endswith(".md"):
        raise ValueError("path must point to a markdown file")


@dataclass(frozen=True)
class FileFingerprint:
    path: str
    mtime_ns: int
    ctime_ns: int
    size: int
    inode: int
    content_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str):
            raise ValueError("fingerprint path must be a string")
        _require_relative_markdown_path(self.path)
        if isinstance(self.mtime_ns, bool) or not isinstance(self.mtime_ns, int):
            raise ValueError("mtime_ns must be an integer")
        integer_values = (self.mtime_ns, self.ctime_ns, self.size, self.inode)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_values):
            raise ValueError("fingerprint metadata must contain integers")
        if any(value < 0 for value in integer_values):
            raise ValueError("fingerprint values must be non-negative")
        if (
            not isinstance(self.content_sha256, str)
            or len(self.content_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.content_sha256)
        ):
            raise ValueError("content_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class MarkdownChunk:
    chunk_id: str
    document_path: str
    title: str
    tags: tuple[str, ...]
    text: str
    line_start: int
    line_end: int

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str)
            for value in (
                self.chunk_id,
                self.document_path,
                self.title,
                self.text,
            )
        ):
            raise ValueError("chunk fields must be strings")
        if not self.chunk_id.strip():
            raise ValueError("chunk_id must not be empty")
        _require_relative_markdown_path(self.document_path)
        if (
            isinstance(self.line_start, bool)
            or not isinstance(self.line_start, int)
            or isinstance(self.line_end, bool)
            or not isinstance(self.line_end, int)
            or self.line_start < 1
            or self.line_end < self.line_start
        ):
            raise ValueError("chunk line range is invalid")
        if not self.text.strip():
            raise ValueError("chunk text must not be empty")
        if not isinstance(self.tags, (tuple, list)) or any(
            not isinstance(tag, str) for tag in self.tags
        ):
            raise ValueError("chunk tags must contain strings")
        object.__setattr__(self, "tags", tuple(self.tags))

    def as_document(self) -> KnowledgeDocument:
        return KnowledgeDocument(
            id=self.chunk_id,
            path=self.document_path,
            title=self.title,
            text=self.text,
            tags=self.tags,
        )


@dataclass(frozen=True)
class IndexedFile:
    fingerprint: FileFingerprint
    document: KnowledgeDocument
    chunks: tuple[MarkdownChunk, ...]

    def __post_init__(self) -> None:
        if self.document.path != self.fingerprint.path:
            raise ValueError("document path must match fingerprint path")
        if self.document.id != self.document.path:
            raise ValueError("cached document id must equal its path")
        if any(chunk.document_path != self.document.path for chunk in self.chunks):
            raise ValueError("chunk path must match document path")
        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunk ids must be unique within a document")


@dataclass(frozen=True)
class VaultIndex:
    vault_path: str
    records: tuple[IndexedFile, ...]
    index_version: int = INDEX_VERSION
    _all_chunks: tuple[MarkdownChunk, ...] = field(init=False, repr=False, compare=False)
    _chunk_lookup: Mapping[str, MarkdownChunk] = field(init=False, repr=False, compare=False)
    _content_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.vault_path:
            raise ValueError("vault_path must not be empty")
        if self.index_version != INDEX_VERSION:
            raise ValueError("unsupported index version")
        paths = [record.document.path for record in self.records]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("index records must have unique sorted paths")
        all_chunks = tuple(chunk for record in self.records for chunk in record.chunks)
        chunk_lookup = {chunk.chunk_id: chunk for chunk in all_chunks}
        if len(chunk_lookup) != len(all_chunks):
            raise ValueError("index chunk ids must be unique")
        digest = hashlib.sha256()
        for record in self.records:
            digest.update(record.document.path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(record.fingerprint.content_sha256.encode("ascii"))
            digest.update(b"\0")
            for chunk in record.chunks:
                digest.update(chunk.chunk_id.encode("utf-8"))
                digest.update(b"\0")
        object.__setattr__(self, "_all_chunks", all_chunks)
        object.__setattr__(self, "_chunk_lookup", MappingProxyType(chunk_lookup))
        object.__setattr__(self, "_content_digest", digest.hexdigest())

    @property
    def documents(self) -> tuple[KnowledgeDocument, ...]:
        return tuple(record.document for record in self.records)

    @property
    def chunks(self) -> tuple[MarkdownChunk, ...]:
        return self._all_chunks

    @property
    def content_digest(self) -> str:
        return self._content_digest

    def document_by_path(self, path: str) -> KnowledgeDocument | None:
        return next(
            (record.document for record in self.records if record.document.path == path),
            None,
        )

    def chunk_by_id(self, chunk_id: str) -> MarkdownChunk | None:
        return self._chunk_lookup.get(chunk_id)
