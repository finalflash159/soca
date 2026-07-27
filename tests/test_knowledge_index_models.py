from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from hashlib import sha256

import pytest

from soca.knowledge.base import KnowledgeDocument
from soca.knowledge.index.models import (
    INDEX_VERSION,
    FileFingerprint,
    IndexedFile,
    MarkdownChunk,
    VaultIndex,
)


def _fingerprint(path: str = "wiki/a.md", text: str = "# A\nalpha") -> FileFingerprint:
    return FileFingerprint(
        path=path,
        mtime_ns=10,
        ctime_ns=11,
        size=len(text.encode("utf-8")),
        inode=12,
        content_sha256=sha256(text.encode("utf-8")).hexdigest(),
    )


def _document(path: str = "wiki/a.md", text: str = "# A\nalpha") -> KnowledgeDocument:
    return KnowledgeDocument(
        id=path,
        path=path,
        title="A",
        text=text,
        tags=("alpha",),
        frontmatter={"kind": "note"},
    )


def _chunk(
    path: str = "wiki/a.md",
    *,
    chunk_id: str = "wiki/a.md#1-2:abc",
    text: str = "# A\nalpha",
) -> MarkdownChunk:
    return MarkdownChunk(
        chunk_id=chunk_id,
        document_path=path,
        title="A",
        tags=("alpha",),
        text=text,
        line_start=1,
        line_end=2,
    )


def _record(path: str = "wiki/a.md", text: str = "# A\nalpha") -> IndexedFile:
    return IndexedFile(
        fingerprint=_fingerprint(path, text),
        document=_document(path, text),
        chunks=(_chunk(path, chunk_id=f"{path}#1-2:abc", text=text),),
    )


def test_index_models_are_frozen_and_deeply_immutable() -> None:
    document = _document()
    fingerprint = _fingerprint()
    chunk = _chunk()
    record = IndexedFile(
        fingerprint=fingerprint,
        document=document,
        chunks=(chunk,),
    )
    index = VaultIndex(vault_path="/vault", records=(record,))

    with pytest.raises(FrozenInstanceError):
        fingerprint.size = 99  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        chunk.text = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        record.chunks = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        index.records = ()  # type: ignore[misc]
    with pytest.raises(TypeError):
        document.frontmatter["kind"] = "changed"  # type: ignore[index]


@pytest.mark.parametrize(
    "path",
    ["", "/absolute.md", "../outside.md", "wiki/../outside.md", "wiki\\a.md", "wiki/a.txt"],
)
def test_fingerprint_rejects_unsafe_or_non_markdown_paths(path: str) -> None:
    with pytest.raises(ValueError, match="path"):
        _fingerprint(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mtime_ns", -1),
        ("ctime_ns", -1),
        ("size", -1),
        ("inode", -1),
        ("mtime_ns", True),
        ("size", 1.5),
    ],
)
def test_fingerprint_rejects_invalid_numeric_metadata(
    field: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "path": "wiki/a.md",
        "mtime_ns": 1,
        "ctime_ns": 2,
        "size": 3,
        "inode": 4,
        "content_sha256": "a" * 64,
    }
    values[field] = value

    with pytest.raises(ValueError):
        FileFingerprint(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("digest", ["", "a" * 63, "A" * 64, "g" * 64])
def test_fingerprint_rejects_invalid_sha256_digest(digest: str) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        FileFingerprint(
            path="wiki/a.md",
            mtime_ns=1,
            ctime_ns=2,
            size=3,
            inode=4,
            content_sha256=digest,
        )


@pytest.mark.parametrize(
    ("line_start", "line_end"),
    [(0, 1), (2, 1), (True, 1), (1, False)],
)
def test_chunk_rejects_invalid_line_ranges(
    line_start: int,
    line_end: int,
) -> None:
    with pytest.raises(ValueError, match="line range"):
        MarkdownChunk(
            chunk_id="wiki/a.md#bad",
            document_path="wiki/a.md",
            title="A",
            tags=(),
            text="alpha",
            line_start=line_start,
            line_end=line_end,
        )


def test_indexed_file_rejects_mismatched_paths_and_duplicate_chunk_ids() -> None:
    document = _document()
    chunk = _chunk()

    with pytest.raises(ValueError, match="fingerprint"):
        IndexedFile(
            fingerprint=_fingerprint("wiki/b.md"),
            document=document,
            chunks=(chunk,),
        )
    with pytest.raises(ValueError, match="chunk path"):
        IndexedFile(
            fingerprint=_fingerprint(),
            document=document,
            chunks=(_chunk("wiki/b.md"),),
        )
    with pytest.raises(ValueError, match="unique"):
        IndexedFile(
            fingerprint=_fingerprint(),
            document=document,
            chunks=(chunk, chunk),
        )


def test_vault_index_requires_supported_version_and_sorted_unique_paths() -> None:
    a = _record("wiki/a.md")
    b = _record("wiki/b.md")

    with pytest.raises(ValueError, match="sorted"):
        VaultIndex(vault_path="/vault", records=(b, a))
    with pytest.raises(ValueError, match="unique"):
        VaultIndex(vault_path="/vault", records=(a, a))
    with pytest.raises(ValueError, match="version"):
        VaultIndex(
            vault_path="/vault",
            records=(a,),
            index_version=INDEX_VERSION + 1,
        )


def test_content_digest_is_stable_and_tracks_searchable_content() -> None:
    original = VaultIndex(vault_path="/vault", records=(_record(),))
    equivalent = VaultIndex(vault_path="/vault", records=(_record(),))
    original_record = original.records[0]
    metadata_only_change = VaultIndex(
        vault_path="/vault",
        records=(
            replace(
                original_record,
                fingerprint=replace(
                    original_record.fingerprint,
                    mtime_ns=99,
                    ctime_ns=100,
                ),
            ),
        ),
    )
    changed_record = _record(text="# A\nalpha changed")
    changed = VaultIndex(vault_path="/vault", records=(changed_record,))

    assert original.content_digest == equivalent.content_digest
    assert original.content_digest == metadata_only_change.content_digest
    assert original.content_digest != changed.content_digest
    assert len(original.content_digest) == 64
    int(original.content_digest, 16)


def test_vault_index_exposes_documents_chunks_and_lookups() -> None:
    record = _record()
    index = VaultIndex(vault_path="/vault", records=(record,))

    assert index.documents == (record.document,)
    assert index.chunks == record.chunks
    assert index.document_by_path("wiki/a.md") is record.document
    assert index.document_by_path("wiki/missing.md") is None
    assert index.chunk_by_id(record.chunks[0].chunk_id) is record.chunks[0]
    assert index.chunk_by_id("missing") is None


def test_document_id_must_equal_path_inside_cached_record() -> None:
    document = replace(_document(), id="different-id")

    with pytest.raises(ValueError, match="id"):
        IndexedFile(
            fingerprint=_fingerprint(),
            document=document,
            chunks=(_chunk(),),
        )
