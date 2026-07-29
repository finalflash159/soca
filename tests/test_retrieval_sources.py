from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.retrieval_sources import (
    DatasetClass,
    load_source_lock,
    write_provision_manifest,
)


def _write_lock(path: Path, *, dataset_class: str = "public_screening") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sources": {
                    "fixture": {
                        "kind": "huggingface",
                        "source": "example/data",
                        "revision": "a" * 40,
                        "dataset_class": dataset_class,
                        "declared_license": "cc-by-sa-4.0",
                        "license_verified": True,
                        "role": "quality",
                        "files": ["corpus.jsonl"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_source_lock_requires_release_eligible_dataset_class(tmp_path: Path) -> None:
    lock_path = tmp_path / "sources.lock.json"
    _write_lock(lock_path, dataset_class="demo_smoke")

    with pytest.raises(ValueError, match="not eligible for quality"):
        load_source_lock(lock_path)


def test_source_lock_requires_immutable_revision(tmp_path: Path) -> None:
    lock_path = tmp_path / "sources.lock.json"
    _write_lock(lock_path)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["sources"]["fixture"]["revision"] = "main"
    lock_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="immutable 40-character revision"):
        load_source_lock(lock_path)


def test_repository_source_lock_has_no_fake_vietnamese_miracl() -> None:
    lock = load_source_lock(
        Path("data/benchmarks/retrieval/sources.lock.json")
    )

    assert all("miracl" not in source.name.lower() for source in lock.sources)
    assert all(
        source.dataset_class in {
            DatasetClass.PUBLIC_SCREENING,
            DatasetClass.SANITIZED_BENCHMARK,
        }
        for source in lock.sources
    )


def test_provision_manifest_hashes_downloaded_bytes(tmp_path: Path) -> None:
    lock_path = tmp_path / "sources.lock.json"
    _write_lock(lock_path)
    lock = load_source_lock(lock_path)
    source_root = tmp_path / "public" / "fixture"
    source_root.mkdir(parents=True)
    (source_root / "corpus.jsonl").write_text('{"id":"1"}\n', encoding="utf-8")

    output = tmp_path / "provisioned-manifest.json"
    manifest = write_provision_manifest(lock, data_root=tmp_path, output=output)

    recorded = manifest["sources"]["fixture"]["files"]["corpus.jsonl"]
    assert recorded["bytes"] == 11
    assert len(recorded["sha256"]) == 64
    assert output.is_file()


def test_provision_manifest_can_cover_only_selected_sources(tmp_path: Path) -> None:
    lock_path = tmp_path / "sources.lock.json"
    _write_lock(lock_path)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["sources"]["other"] = {
        **payload["sources"]["fixture"],
        "revision": "b" * 40,
    }
    lock_path.write_text(json.dumps(payload), encoding="utf-8")
    source_root = tmp_path / "public" / "fixture"
    source_root.mkdir(parents=True)
    (source_root / "corpus.jsonl").write_text('{"id":"1"}\n', encoding="utf-8")

    manifest = write_provision_manifest(
        load_source_lock(lock_path),
        data_root=tmp_path,
        output=tmp_path / "manifest.json",
        selected={"fixture"},
    )

    assert set(manifest["sources"]) == {"fixture"}


def test_partial_provision_preserves_existing_source_records(tmp_path: Path) -> None:
    lock_path = tmp_path / "sources.lock.json"
    _write_lock(lock_path)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["sources"]["other"] = {
        **payload["sources"]["fixture"],
        "revision": "b" * 40,
    }
    lock_path.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "manifest.json"
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_lock_sha256": "old",
                "sources": {"other": {"revision": "b" * 40, "files": {}}},
            }
        ),
        encoding="utf-8",
    )
    source_root = tmp_path / "public" / "fixture"
    source_root.mkdir(parents=True)
    (source_root / "corpus.jsonl").write_text('{"id":"1"}\n', encoding="utf-8")

    manifest = write_provision_manifest(
        load_source_lock(lock_path),
        data_root=tmp_path,
        output=output,
        selected={"fixture"},
    )

    assert set(manifest["sources"]) == {"fixture", "other"}
