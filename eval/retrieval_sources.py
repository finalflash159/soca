from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class DatasetClass(StrEnum):
    PUBLIC_SCREENING = "public_screening"
    SANITIZED_BENCHMARK = "sanitized_benchmark"
    PRIVATE_RELEASE = "private_release"


@dataclass(frozen=True)
class RetrievalSource:
    name: str
    kind: str
    source: str
    revision: str
    dataset_class: DatasetClass
    declared_license: str
    license_verified: bool
    role: str
    destination: Path
    files: tuple[str, ...]
    include_globs: tuple[str, ...]
    sparse_paths: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalSourceLock:
    path: Path
    schema_version: int
    sources: tuple[RetrievalSource, ...]


def _required_string(payload: dict[str, Any], field: str, *, source: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: {field} must be a non-empty string")
    return value.strip()


def _string_tuple(payload: dict[str, Any], field: str, *, source: str) -> tuple[str, ...]:
    value = payload.get(field, [])
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{source}: {field} must contain non-empty strings")
    return tuple(item.strip() for item in value)


def load_source_lock(path: Path) -> RetrievalSourceLock:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("retrieval source lock must use schema_version 1")
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, dict) or not raw_sources:
        raise ValueError("retrieval source lock requires sources")

    sources: list[RetrievalSource] = []
    for name, raw_source in raw_sources.items():
        if not isinstance(name, str) or not name or not isinstance(raw_source, dict):
            raise ValueError("retrieval source entries must be named objects")
        dataset_class_value = _required_string(
            raw_source, "dataset_class", source=name
        )
        try:
            dataset_class = DatasetClass(dataset_class_value)
        except ValueError as exc:
            raise ValueError(
                f"{name}: dataset class {dataset_class_value!r} is not eligible for quality"
            ) from exc
        if dataset_class == DatasetClass.PRIVATE_RELEASE:
            raise ValueError(f"{name}: private release data must not be in the public source lock")

        kind = _required_string(raw_source, "kind", source=name)
        if kind not in {"git", "huggingface"}:
            raise ValueError(f"{name}: unsupported source kind {kind!r}")

        revision = _required_string(raw_source, "revision", source=name)
        if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
            raise ValueError(f"{name}: revision must be an immutable 40-character revision")

        destination_value = raw_source.get("destination")
        if destination_value is None:
            parent = (
                "sanitized"
                if dataset_class == DatasetClass.SANITIZED_BENCHMARK
                else "public"
            )
            destination = Path(parent) / name
        elif isinstance(destination_value, str) and destination_value.strip():
            destination = Path(destination_value)
        else:
            raise ValueError(f"{name}: destination must be a relative path")
        if destination.is_absolute() or ".." in destination.parts:
            raise ValueError(f"{name}: destination must stay under the retrieval data root")

        files = _string_tuple(raw_source, "files", source=name)
        include_globs = _string_tuple(raw_source, "include_globs", source=name)
        if not files and not include_globs:
            raise ValueError(f"{name}: files or include_globs must be declared")
        sources.append(
            RetrievalSource(
                name=name,
                kind=kind,
                source=_required_string(raw_source, "source", source=name),
                revision=revision,
                dataset_class=dataset_class,
                declared_license=_required_string(
                    raw_source, "declared_license", source=name
                ),
                license_verified=raw_source.get("license_verified") is True,
                role=_required_string(raw_source, "role", source=name),
                destination=destination,
                files=files,
                include_globs=include_globs,
                sparse_paths=_string_tuple(raw_source, "sparse_paths", source=name),
            )
        )
    return RetrievalSourceLock(path=path, schema_version=1, sources=tuple(sources))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_files(source: RetrievalSource, root: Path) -> tuple[Path, ...]:
    selected = {root / relative for relative in source.files}
    for pattern in source.include_globs:
        selected.update(path for path in root.glob(pattern) if path.is_file())
    return tuple(sorted(selected))


def write_provision_manifest(
    lock: RetrievalSourceLock,
    *,
    data_root: Path,
    output: Path,
    selected: set[str] | None = None,
) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    if selected is not None and output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        existing_sources = existing.get("sources") if isinstance(existing, dict) else None
        if isinstance(existing_sources, dict):
            sources.update(existing_sources)
    for source in lock.sources:
        if selected is not None and source.name not in selected:
            continue
        root = data_root / source.destination
        source_files = _source_files(source, root)
        if not source_files:
            raise FileNotFoundError(
                f"{source.name}: no provisioned files matched the source lock"
            )
        files: dict[str, dict[str, Any]] = {}
        for path in source_files:
            if not path.is_file():
                raise FileNotFoundError(
                    f"{source.name}: provisioned file is missing: {path}"
                )
            relative = path.relative_to(root).as_posix()
            files[relative] = {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        sources[source.name] = {
            "dataset_class": source.dataset_class.value,
            "role": source.role,
            "revision": source.revision,
            "declared_license": source.declared_license,
            "license_verified": source.license_verified,
            "files": files,
        }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source_lock_sha256": file_sha256(lock.path),
        "sources": sources,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return manifest
