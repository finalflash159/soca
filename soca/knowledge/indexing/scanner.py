from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from soca.knowledge.base import KnowledgeDocument
from soca.knowledge.index.chunker import chunk_markdown
from soca.knowledge.index.models import FileFingerprint, IndexedFile, VaultIndex

LOGGER = logging.getLogger(__name__)


class VaultReader(Protocol):
    root: Path

    def iter_paths(self) -> tuple[str, ...]: ...

    def read(self, path: str) -> KnowledgeDocument: ...


@dataclass(frozen=True)
class ScanReport:
    index: VaultIndex
    scanned: int
    changed: int
    added: int
    removed: int
    metadata_only: int


def _probe_file(reader: VaultReader, relative_path: str) -> tuple[str, int, int, int, int]:
    root = reader.root.expanduser().resolve()
    candidate = root / relative_path
    current = root
    for part in Path(relative_path).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlink paths are excluded from the catalog")
    path = candidate.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("vault reader returned a path outside its root") from exc
    stat_result = path.stat()
    if not path.is_file():
        raise ValueError("vault path is not a regular file")
    return (
        path.relative_to(root).as_posix(),
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
        stat_result.st_size,
        stat_result.st_ino,
    )


def _fingerprint(
    path: str,
    mtime_ns: int,
    ctime_ns: int,
    size: int,
    inode: int,
    document: KnowledgeDocument,
) -> FileFingerprint:
    return FileFingerprint(
        path=path,
        mtime_ns=mtime_ns,
        ctime_ns=ctime_ns,
        size=size,
        inode=inode,
        content_sha256=hashlib.sha256(document.text.encode("utf-8")).hexdigest(),
    )


def scan_vault(
    reader: VaultReader,
    *,
    previous: VaultIndex | None = None,
    verify_content: bool = False,
) -> ScanReport:
    expected_root = str(reader.root.expanduser().resolve())
    if previous is not None and previous.vault_path != expected_root:
        raise ValueError("previous index belongs to a different vault")
    old_by_path = (
        {record.fingerprint.path: record for record in previous.records}
        if previous is not None
        else {}
    )
    records: list[IndexedFile] = []
    changed = added = metadata_only = 0
    paths = tuple(sorted(set(reader.iter_paths())))
    for relative_path in paths:
        try:
            path, mtime_ns, ctime_ns, size, inode = _probe_file(reader, relative_path)
        except (OSError, ValueError) as exc:
            LOGGER.debug("Skipping unavailable knowledge path %s: %s", relative_path, exc)
            continue
        old_record = old_by_path.get(path)
        if (
            old_record is not None
            and not verify_content
            and old_record.fingerprint.mtime_ns == mtime_ns
            and old_record.fingerprint.ctime_ns == ctime_ns
            and old_record.fingerprint.size == size
            and old_record.fingerprint.inode == inode
        ):
            records.append(old_record)
            continue
        try:
            document = reader.read(path)
            fingerprint = _fingerprint(path, mtime_ns, ctime_ns, size, inode, document)
            if old_record is not None and old_record.fingerprint.content_sha256 == fingerprint.content_sha256:
                records.append(
                    IndexedFile(
                        fingerprint=fingerprint,
                        document=old_record.document,
                        chunks=old_record.chunks,
                    )
                )
                metadata_only += 1
                continue
            chunks = chunk_markdown(document)
        except (OSError, ValueError) as exc:
            LOGGER.warning("Skipping unreadable knowledge path %s: %s", path, exc)
            continue
        records.append(IndexedFile(fingerprint=fingerprint, document=document, chunks=chunks))
        changed += 1
        if old_record is None:
            added += 1

    removed = len(set(old_by_path) - {record.fingerprint.path for record in records})
    index = VaultIndex(vault_path=expected_root, records=tuple(sorted(records, key=lambda item: item.document.path)))
    return ScanReport(
        index=index,
        scanned=len(records),
        changed=changed,
        added=added,
        removed=removed,
        metadata_only=metadata_only,
    )
