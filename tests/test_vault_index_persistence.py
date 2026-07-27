from __future__ import annotations

import json
import logging
import os
import stat
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from soca.knowledge.base import KnowledgeDocument
from soca.knowledge.index.models import (
    INDEX_VERSION,
    FileFingerprint,
    IndexedFile,
    MarkdownChunk,
    VaultIndex,
)
from soca.knowledge.index.persistence import (
    InvalidManifestError,
    index_from_json,
    index_to_json,
    load_index,
    manifest_path_for,
    save_index,
)


def _index(vault: Path) -> VaultIndex:
    text = "# Alpha\nalpha body"
    path = "wiki/alpha.md"
    document = KnowledgeDocument(
        id=path,
        path=path,
        title="Alpha",
        text=text,
        tags=("alpha",),
        frontmatter={"kind": "note"},
    )
    fingerprint = FileFingerprint(
        path=path,
        mtime_ns=10,
        ctime_ns=11,
        size=len(text.encode("utf-8")),
        inode=12,
        content_sha256=sha256(text.encode("utf-8")).hexdigest(),
    )
    chunk = MarkdownChunk(
        chunk_id=f"{path}#1-2:abc",
        document_path=path,
        title="Alpha",
        tags=("alpha",),
        text=text,
        line_start=1,
        line_end=2,
    )
    return VaultIndex(
        vault_path=str(vault.resolve()),
        records=(
            IndexedFile(
                fingerprint=fingerprint,
                document=document,
                chunks=(chunk,),
            ),
        ),
    )


def _write_payload(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_manifest_round_trip_preserves_the_complete_index(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    index = _index(vault)
    manifest = manifest_path_for(tmp_path / "index-home", vault)

    save_index(manifest, index)

    assert (
        load_index(
            manifest,
            expected_vault_path=str(vault.resolve()),
        )
        == index
    )
    assert (
        index_from_json(
            index_to_json(index),
            expected_vault_path=str(vault.resolve()),
        )
        == index
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits are unavailable")
def test_saved_manifest_uses_private_directory_and_file_modes(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    manifest = manifest_path_for(tmp_path / "index-home", vault)

    save_index(manifest, _index(vault))

    assert stat.S_IMODE(manifest.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(manifest.stat().st_mode) == 0o600


def test_load_returns_cache_miss_for_corrupt_json(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{not valid json", encoding="utf-8")

    assert load_index(manifest, expected_vault_path="/vault") is None


def test_missing_manifest_is_a_silent_cache_miss(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    manifest = tmp_path / "missing.json"

    with caplog.at_level(
        logging.WARNING,
        logger="soca.knowledge.index.persistence",
    ):
        assert load_index(manifest, expected_vault_path="/vault") is None

    assert caplog.records == []


def test_load_returns_cache_miss_for_invalid_schema(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    manifest = tmp_path / "manifest.json"
    payload = index_to_json(_index(vault))
    payload["records"] = {"not": "a list"}
    _write_payload(manifest, payload)

    assert (
        load_index(
            manifest,
            expected_vault_path=str(vault.resolve()),
        )
        is None
    )


def test_load_returns_cache_miss_for_wrong_root_or_version(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    manifest = tmp_path / "manifest.json"
    payload = index_to_json(_index(vault))
    payload["index_version"] = INDEX_VERSION + 1
    _write_payload(manifest, payload)

    assert (
        load_index(
            manifest,
            expected_vault_path=str(vault.resolve()),
        )
        is None
    )

    payload["index_version"] = INDEX_VERSION
    _write_payload(manifest, payload)
    assert load_index(manifest, expected_vault_path="/different-vault") is None


def test_duplicate_document_path_is_rejected_as_invalid_manifest(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    payload = index_to_json(_index(vault))
    records = payload["records"]
    assert isinstance(records, list)
    records.append(deepcopy(records[0]))

    with pytest.raises(InvalidManifestError, match="unique"):
        index_from_json(
            payload,
            expected_vault_path=str(vault.resolve()),
        )


@pytest.mark.parametrize(
    ("container", "field", "value"),
    [
        ("chunk", "line_end", 0),
        ("fingerprint", "path", "wiki/different.md"),
    ],
)
def test_invalid_nested_record_becomes_a_cache_miss(
    tmp_path: Path,
    container: str,
    field: str,
    value: object,
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    manifest = tmp_path / "manifest.json"
    payload = index_to_json(_index(vault))
    records = payload["records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    if container == "chunk":
        chunks = record["chunks"]
        assert isinstance(chunks, list)
        target = chunks[0]
    else:
        target = record["fingerprint"]
    assert isinstance(target, dict)
    target[field] = value
    _write_payload(manifest, payload)

    assert (
        load_index(
            manifest,
            expected_vault_path=str(vault.resolve()),
        )
        is None
    )
