from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
from pathlib import Path
from typing import Any
from uuid import uuid4

from soca.knowledge.base import KnowledgeDocument
from soca.knowledge.index.models import (
    INDEX_VERSION,
    FileFingerprint,
    IndexedFile,
    MarkdownChunk,
    VaultIndex,
)

LOGGER = logging.getLogger(__name__)
MAX_MANIFEST_BYTES = 256 * 1024 * 1024


class InvalidManifestError(ValueError):
    pass


def default_index_home(vault: Path | None = None) -> Path:
    if vault is not None:
        return vault.expanduser().resolve() / ".soca" / "knowledge_index"
    configured = os.environ.get("XDG_CONFIG_HOME")
    if configured:
        base = Path(configured).expanduser()
        if not base.is_absolute():
            raise ValueError("XDG_CONFIG_HOME must be absolute")
    else:
        base = Path.home() / ".config"
    return base / "soca" / "knowledge_index"


def manifest_path_for(index_home: Path, vault_root: Path) -> Path:
    resolved = str(vault_root.expanduser().resolve())
    key = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
    return index_home.expanduser() / key / "manifest.json"


def _require_dict(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise InvalidManifestError(f"{field} must be an object")
    return value


def _require_list(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise InvalidManifestError(f"{field} must be a list")
    return value


def _require_str(value: object, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise InvalidManifestError(f"{field} must be a string")
    return value


def _require_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InvalidManifestError(f"{field} must be an integer >= {minimum}")
    return value


def _document_to_json(document: KnowledgeDocument) -> dict[str, object]:
    return {
        "id": document.id,
        "path": document.path,
        "title": document.title,
        "text": document.text,
        "tags": list(document.tags),
        "frontmatter": dict(document.frontmatter),
    }


def _chunk_to_json(chunk: MarkdownChunk) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_path": chunk.document_path,
        "title": chunk.title,
        "tags": list(chunk.tags),
        "text": chunk.text,
        "line_start": chunk.line_start,
        "line_end": chunk.line_end,
    }


def index_to_json(index: VaultIndex) -> dict[str, object]:
    return {
        "index_version": index.index_version,
        "vault_path": index.vault_path,
        "records": [
            {
                "fingerprint": {
                    "path": record.fingerprint.path,
                    "mtime_ns": record.fingerprint.mtime_ns,
                    "ctime_ns": record.fingerprint.ctime_ns,
                    "size": record.fingerprint.size,
                    "inode": record.fingerprint.inode,
                    "content_sha256": record.fingerprint.content_sha256,
                },
                "document": _document_to_json(record.document),
                "chunks": [_chunk_to_json(chunk) for chunk in record.chunks],
            }
            for record in index.records
        ],
    }


def _parse_string_tuple(value: object, field: str) -> tuple[str, ...]:
    values = _require_list(value, field)
    if any(not isinstance(item, str) for item in values):
        raise InvalidManifestError(f"{field} must contain only strings")
    return tuple(values)


def _parse_frontmatter(value: object) -> dict[str, str]:
    payload = _require_dict(value, "frontmatter")
    if any(not isinstance(item, str) for item in payload.values()):
        raise InvalidManifestError("frontmatter values must be strings")
    return {key: item for key, item in payload.items()}


def _parse_document(value: object) -> KnowledgeDocument:
    payload = _require_dict(value, "document")
    return KnowledgeDocument(
        id=_require_str(payload.get("id"), "document.id"),
        path=_require_str(payload.get("path"), "document.path"),
        title=_require_str(payload.get("title"), "document.title", allow_empty=True),
        text=_require_str(payload.get("text"), "document.text", allow_empty=True),
        tags=_parse_string_tuple(payload.get("tags"), "document.tags"),
        frontmatter=_parse_frontmatter(payload.get("frontmatter")),
    )


def _parse_chunk(value: object) -> MarkdownChunk:
    payload = _require_dict(value, "chunk")
    return MarkdownChunk(
        chunk_id=_require_str(payload.get("chunk_id"), "chunk.chunk_id"),
        document_path=_require_str(payload.get("document_path"), "chunk.document_path"),
        title=_require_str(payload.get("title"), "chunk.title", allow_empty=True),
        tags=_parse_string_tuple(payload.get("tags"), "chunk.tags"),
        text=_require_str(payload.get("text"), "chunk.text"),
        line_start=_require_int(payload.get("line_start"), "chunk.line_start", minimum=1),
        line_end=_require_int(payload.get("line_end"), "chunk.line_end", minimum=1),
    )


def _parse_record(value: object) -> IndexedFile:
    payload = _require_dict(value, "record")
    raw_fingerprint = _require_dict(payload.get("fingerprint"), "fingerprint")
    fingerprint = FileFingerprint(
        path=_require_str(raw_fingerprint.get("path"), "fingerprint.path"),
        mtime_ns=_require_int(raw_fingerprint.get("mtime_ns"), "fingerprint.mtime_ns"),
        ctime_ns=_require_int(raw_fingerprint.get("ctime_ns"), "fingerprint.ctime_ns"),
        size=_require_int(raw_fingerprint.get("size"), "fingerprint.size"),
        inode=_require_int(raw_fingerprint.get("inode"), "fingerprint.inode"),
        content_sha256=_require_str(
            raw_fingerprint.get("content_sha256"),
            "fingerprint.content_sha256",
        ),
    )
    return IndexedFile(
        fingerprint=fingerprint,
        document=_parse_document(payload.get("document")),
        chunks=tuple(_parse_chunk(item) for item in _require_list(payload.get("chunks"), "chunks")),
    )


def index_from_json(value: object, *, expected_vault_path: str) -> VaultIndex:
    payload = _require_dict(value, "manifest")
    version = _require_int(payload.get("index_version"), "index_version")
    if version != INDEX_VERSION:
        raise InvalidManifestError("index version mismatch")
    vault_path = _require_str(payload.get("vault_path"), "vault_path")
    if vault_path != expected_vault_path:
        raise InvalidManifestError("vault path mismatch")
    try:
        records = tuple(
            _parse_record(item) for item in _require_list(payload.get("records"), "records")
        )
        return VaultIndex(
            vault_path=vault_path,
            records=records,
            index_version=version,
        )
    except InvalidManifestError:
        raise
    except ValueError as exc:
        raise InvalidManifestError(str(exc)) from exc


def load_index(path: Path, *, expected_vault_path: str) -> VaultIndex | None:
    try:
        metadata = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > MAX_MANIFEST_BYTES
        ):
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return index_from_json(payload, expected_vault_path=expected_vault_path)
    except FileNotFoundError:
        return None
    except (
        InvalidManifestError,
        json.JSONDecodeError,
        OSError,
        UnicodeDecodeError,
    ):
        LOGGER.warning("Ignoring invalid local knowledge index cache", exc_info=True)
        return None


def ensure_private_directory(path: Path) -> None:
    if path.is_symlink():
        raise OSError("cache directory must not be a symlink")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise OSError("cache path must be a directory")
    if os.name == "posix":
        os.chmod(path, 0o700)


def fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_index(path: Path, index: VaultIndex) -> None:
    ensure_private_directory(path.parent)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    payload = json.dumps(index_to_json(index), ensure_ascii=False, sort_keys=True)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            os.chmod(path, 0o600)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
