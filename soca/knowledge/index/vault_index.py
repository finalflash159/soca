from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from soca.knowledge.base import KnowledgeDocument
from soca.knowledge.index.chunker import chunk_markdown
from soca.knowledge.index.models import FileFingerprint, IndexedFile, VaultIndex
from soca.knowledge.index.persistence import (
    default_index_home,
    load_index,
    manifest_path_for,
    save_index,
)

LOGGER = logging.getLogger(__name__)


class VaultReader(Protocol):
    root: Path

    def iter_paths(self) -> tuple[str, ...]: ...

    def read(self, path: str) -> KnowledgeDocument: ...


def _resolved_vault_path(root: Path) -> str:
    return str(root.expanduser().resolve())


@dataclass(frozen=True)
class _FileProbe:
    path: str
    mtime_ns: int
    ctime_ns: int
    size: int
    inode: int

    def matches(self, fingerprint: FileFingerprint) -> bool:
        return (
            self.path == fingerprint.path
            and self.mtime_ns == fingerprint.mtime_ns
            and self.ctime_ns == fingerprint.ctime_ns
            and self.size == fingerprint.size
            and self.inode == fingerprint.inode
        )

    def fingerprint(self, document: KnowledgeDocument) -> FileFingerprint:
        digest = hashlib.sha256(document.text.encode("utf-8")).hexdigest()
        return FileFingerprint(
            path=self.path,
            mtime_ns=self.mtime_ns,
            ctime_ns=self.ctime_ns,
            size=self.size,
            inode=self.inode,
            content_sha256=digest,
        )


def _probe_file(reader: VaultReader, relative_path: str) -> _FileProbe:
    root = reader.root.expanduser().resolve()
    path = (root / relative_path).resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("vault reader returned a path outside its root") from exc
    stat_result = path.stat()
    return _FileProbe(
        path=path.relative_to(root).as_posix(),
        mtime_ns=stat_result.st_mtime_ns,
        ctime_ns=stat_result.st_ctime_ns,
        size=stat_result.st_size,
        inode=stat_result.st_ino,
    )


class VaultIndexStore:
    def __init__(self, *, index_home: Path | None = None) -> None:
        self.index_home = (index_home or default_index_home()).expanduser()

    def manifest_path_for(self, vault_root: Path) -> Path:
        return manifest_path_for(self.index_home, vault_root)

    def load(self, vault_root: Path) -> VaultIndex | None:
        expected = _resolved_vault_path(vault_root)
        return load_index(
            self.manifest_path_for(vault_root),
            expected_vault_path=expected,
        )

    def persist(self, index: VaultIndex) -> None:
        path = manifest_path_for(self.index_home, Path(index.vault_path))
        save_index(path, index)


class VaultIndexer:
    def __init__(self, reader: VaultReader, store: VaultIndexStore) -> None:
        self.reader = reader
        self.store = store

    def refresh(
        self,
        previous: VaultIndex | None = None,
        *,
        verify_content: bool = False,
    ) -> VaultIndex:
        expected_vault_path = _resolved_vault_path(self.reader.root)
        if previous is not None and previous.vault_path != expected_vault_path:
            raise ValueError("previous index belongs to a different vault")
        cached = previous if previous is not None else self.store.load(self.reader.root)
        cached_by_path = (
            {record.document.path: record for record in cached.records}
            if cached is not None
            else {}
        )

        records: list[IndexedFile] = []
        for relative_path in self.reader.iter_paths():
            probe = _probe_file(self.reader, relative_path)
            old_record = cached_by_path.get(probe.path)
            if (
                old_record is not None
                and probe.matches(old_record.fingerprint)
                and not verify_content
            ):
                records.append(old_record)
                continue

            document = self.reader.read(probe.path)
            fingerprint = probe.fingerprint(document)
            if (
                old_record is not None
                and old_record.fingerprint.content_sha256 == fingerprint.content_sha256
            ):
                records.append(
                    old_record
                    if old_record.fingerprint == fingerprint
                    else IndexedFile(
                        fingerprint=fingerprint,
                        document=old_record.document,
                        chunks=old_record.chunks,
                    )
                )
                continue
            records.append(
                IndexedFile(
                    fingerprint=fingerprint,
                    document=document,
                    chunks=chunk_markdown(document),
                )
            )

        index = VaultIndex(
            vault_path=expected_vault_path,
            records=tuple(records),
        )
        if cached is not None and index == cached:
            return cached

        try:
            self.store.persist(index)
        except OSError:
            LOGGER.warning(
                "Knowledge index persistence failed; continuing with the in-memory index",
                exc_info=True,
            )
        return index
