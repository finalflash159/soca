from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from importlib.resources import files as resource_files
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

ARTIFACT_MANIFEST_SCHEMA_VERSION = 1
QWEN_ASR_MODEL_ROOT_ENV = "SOCA_QWEN_ASR_MODEL_ROOT"
_HEX_DIGITS = frozenset("0123456789abcdef")


class QwenArtifactError(RuntimeError):
    """Base failure for Qwen ASR artifact identity and storage."""


class QwenArtifactManifestError(QwenArtifactError):
    """An artifact manifest contains malformed or mutable identity."""


class QwenArtifactSchemaError(QwenArtifactManifestError):
    """An artifact manifest uses an unsupported schema."""


class QwenArtifactRoleError(QwenArtifactError):
    """The selected artifact does not have the required operational role."""


class QwenArtifactPermissionError(QwenArtifactError):
    """An artifact receipt is not a private regular file."""


class QwenArtifactPathError(QwenArtifactError):
    """A local artifact path is missing, indirect or not a directory."""


class ArtifactRole(StrEnum):
    RELEASE = "release"
    REFERENCE = "reference"


def _is_lower_hex(value: str, length: int) -> bool:
    return len(value) == length and all(character in _HEX_DIGITS for character in value)


def _validate_stable_key(value: str) -> None:
    if not value or value[0] == "_" or value[-1] == "_":
        raise QwenArtifactManifestError("artifact key must be a stable lowercase identifier")
    if any(not (character.isascii() and (character.islower() or character.isdigit() or character == "_")) for character in value):
        raise QwenArtifactManifestError("artifact key must be a stable lowercase identifier")


@dataclass(frozen=True, slots=True)
class ArtifactSource:
    repo_id: str
    revision: str

    def __post_init__(self) -> None:
        components = self.repo_id.split("/")
        if len(components) != 2 or any(not component or component in {".", ".."} for component in components):
            raise QwenArtifactManifestError("artifact repo_id must be an owner/repository pair")
        if not _is_lower_hex(self.revision, 40):
            raise QwenArtifactManifestError("artifact revision must be an immutable commit SHA")

    def to_dict(self) -> dict[str, str]:
        return {"repo_id": self.repo_id, "revision": self.revision}


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        candidate = PurePosixPath(self.path)
        if (
            not self.path
            or "\\" in self.path
            or candidate.is_absolute()
            or candidate.as_posix() != self.path
            or any(part in {"", ".", ".."} for part in candidate.parts)
        ):
            raise QwenArtifactManifestError("artifact file path must be normalized and relative")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 1:
            raise QwenArtifactManifestError("artifact file size must be a positive integer")
        if not _is_lower_hex(self.sha256, 64):
            raise QwenArtifactManifestError("artifact file sha256 must be lowercase hexadecimal")

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class QwenASRArtifactSpec:
    key: str
    role: ArtifactRole
    upstream: ArtifactSource
    mirror: ArtifactSource | None
    files: tuple[ArtifactFile, ...]
    license: str
    device: str
    dtype: str
    runtime_lock_digest: str | None
    context_policy_digest: str | None
    minimum_protocol_version: int

    def __post_init__(self) -> None:
        _validate_stable_key(self.key)
        if not isinstance(self.role, ArtifactRole):
            raise QwenArtifactManifestError("artifact role must be typed")
        if not isinstance(self.upstream, ArtifactSource) or (
            self.mirror is not None and not isinstance(self.mirror, ArtifactSource)
        ):
            raise QwenArtifactManifestError("artifact sources must be typed")
        if not self.files:
            raise QwenArtifactManifestError("artifact file manifest must not be empty")
        if any(not isinstance(file, ArtifactFile) for file in self.files):
            raise QwenArtifactManifestError("artifact files must be typed")
        paths = [file.path for file in self.files]
        if len(paths) != len(set(paths)) or paths != sorted(paths):
            raise QwenArtifactManifestError("artifact files must be unique and path-sorted")
        if not self.license.strip() or not self.device.strip() or not self.dtype.strip():
            raise QwenArtifactManifestError("license, device and dtype must not be empty")
        for name, digest in (
            ("runtime_lock_digest", self.runtime_lock_digest),
            ("context_policy_digest", self.context_policy_digest),
        ):
            if digest is not None and not _is_lower_hex(digest, 64):
                raise QwenArtifactManifestError(f"{name} must be a SHA-256 digest or null")
        if (
            isinstance(self.minimum_protocol_version, bool)
            or not isinstance(self.minimum_protocol_version, int)
            or self.minimum_protocol_version < 1
        ):
            raise QwenArtifactManifestError("minimum_protocol_version must be positive")

    @property
    def canonical_json(self) -> str:
        return canonical_manifest_json(self.to_manifest_dict())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @property
    def total_bytes(self) -> int:
        return sum(file.size for file in self.files)

    def file(self, path: str) -> ArtifactFile:
        for entry in self.files:
            if entry.path == path:
                return entry
        raise QwenArtifactManifestError(f"artifact file is not declared: {path}")

    def model_path(self, data_root: Path | None = None) -> Path:
        root = default_asr_model_root() if data_root is None else _absolute_path(data_root)
        return root / self.key / self.upstream.revision

    def receipt_path(self, data_root: Path | None = None) -> Path:
        root = default_asr_model_root() if data_root is None else _absolute_path(data_root)
        return root / "receipts" / f"{self.key}.json"

    def to_manifest_dict(self) -> dict[str, object]:
        return {
            "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "key": self.key,
            "role": self.role.value,
            "upstream": self.upstream.to_dict(),
            "mirror": self.mirror.to_dict() if self.mirror is not None else None,
            "files": [file.to_dict() for file in self.files],
            "license": self.license,
            "device": self.device,
            "dtype": self.dtype,
            "runtime_lock_digest": self.runtime_lock_digest,
            "context_policy_digest": self.context_policy_digest,
            "minimum_protocol_version": self.minimum_protocol_version,
        }


def canonical_manifest_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _absolute_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        raise QwenArtifactManifestError("Qwen ASR data root must be absolute")
    if ".." in expanded.parts:
        raise QwenArtifactManifestError("Qwen ASR data root must not contain traversal")
    return expanded


def default_asr_model_root() -> Path:
    from soca.model_paths import default_model_root

    configured = os.environ.get(QWEN_ASR_MODEL_ROOT_ENV, "").strip()
    if configured:
        try:
            return _absolute_path(Path(configured))
        except QwenArtifactManifestError:
            raise
    try:
        return default_model_root() / "asr"
    except ValueError as exc:
        raise QwenArtifactManifestError(str(exc)) from exc


def validate_private_receipt(path: Path) -> None:
    target = _absolute_path(path)
    for ancestor in (target, *target.parents):
        if ancestor.is_symlink():
            raise QwenArtifactPermissionError("artifact receipt path must not contain a symlink")
    try:
        metadata = target.lstat()
    except OSError as exc:
        raise QwenArtifactPermissionError("artifact receipt is not readable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise QwenArtifactPermissionError("artifact receipt must be a regular file")
    mode = stat.S_IMODE(metadata.st_mode)
    if mode & 0o077:
        raise QwenArtifactPermissionError("artifact receipt permissions must be private")
    if not mode & stat.S_IRUSR:
        raise QwenArtifactPermissionError("artifact receipt must be readable by owner")


def validate_local_model_directory(path: Path) -> Path:
    target = path.expanduser()
    if not target.is_absolute() or ".." in target.parts:
        raise QwenArtifactPathError(
            "Qwen model path must be absolute without traversal"
        )
    try:
        for component in (target, *target.parents):
            metadata = component.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise QwenArtifactPathError(
                    f"Qwen model path contains a symlink: {component}"
                )
        if not stat.S_ISDIR(target.lstat().st_mode):
            raise QwenArtifactPathError(
                "Qwen model path must be a local directory"
            )
    except FileNotFoundError as exc:
        raise QwenArtifactPathError("Qwen model path does not exist") from exc
    except OSError as exc:
        raise QwenArtifactPathError("Qwen model path cannot be inspected") from exc
    return target


def _require_object(payload: object, name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise QwenArtifactManifestError(f"{name} must be an object")
    return payload


def _require_exact_keys(payload: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    keys = frozenset(payload)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise QwenArtifactManifestError(f"{name} fields mismatch: missing={missing}, extra={extra}")


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise QwenArtifactManifestError(f"{key} must be a non-empty string")
    return value


def _decode_source(payload: object, name: str) -> ArtifactSource:
    source = _require_object(payload, name)
    _require_exact_keys(source, frozenset({"repo_id", "revision"}), name)
    return ArtifactSource(
        repo_id=_required_string(source, "repo_id"),
        revision=_required_string(source, "revision"),
    )


def _decode_file(payload: object) -> ArtifactFile:
    item = _require_object(payload, "artifact file")
    _require_exact_keys(item, frozenset({"path", "size", "sha256"}), "artifact file")
    size = item.get("size")
    if isinstance(size, bool) or not isinstance(size, int):
        raise QwenArtifactManifestError("artifact file size must be an integer")
    return ArtifactFile(
        path=_required_string(item, "path"),
        size=size,
        sha256=_required_string(item, "sha256"),
    )


def _optional_digest(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise QwenArtifactManifestError(f"{key} must be a string or null")
    return value


def decode_artifact_manifest(payload: object) -> QwenASRArtifactSpec:
    manifest = _require_object(payload, "artifact manifest")
    expected_keys = frozenset(
        {
            "schema_version",
            "key",
            "role",
            "upstream",
            "mirror",
            "files",
            "license",
            "device",
            "dtype",
            "runtime_lock_digest",
            "context_policy_digest",
            "minimum_protocol_version",
        }
    )
    _require_exact_keys(manifest, expected_keys, "artifact manifest")
    schema_version = manifest.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or schema_version != ARTIFACT_MANIFEST_SCHEMA_VERSION
    ):
        raise QwenArtifactSchemaError("unsupported Qwen ASR artifact manifest schema")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise QwenArtifactManifestError("files must be an array")
    raw_protocol = manifest.get("minimum_protocol_version")
    if isinstance(raw_protocol, bool) or not isinstance(raw_protocol, int):
        raise QwenArtifactManifestError("minimum_protocol_version must be an integer")
    raw_mirror = manifest.get("mirror")
    try:
        role = ArtifactRole(_required_string(manifest, "role"))
    except ValueError as exc:
        raise QwenArtifactManifestError("artifact role is not supported") from exc
    return QwenASRArtifactSpec(
        key=_required_string(manifest, "key"),
        role=role,
        upstream=_decode_source(manifest.get("upstream"), "upstream"),
        mirror=_decode_source(raw_mirror, "mirror") if raw_mirror is not None else None,
        files=tuple(_decode_file(item) for item in raw_files),
        license=_required_string(manifest, "license"),
        device=_required_string(manifest, "device"),
        dtype=_required_string(manifest, "dtype"),
        runtime_lock_digest=_optional_digest(manifest, "runtime_lock_digest"),
        context_policy_digest=_optional_digest(manifest, "context_policy_digest"),
        minimum_protocol_version=raw_protocol,
    )


def _load_builtin_manifest(filename: str) -> QwenASRArtifactSpec:
    resource = resource_files("soca.asr.artifacts").joinpath(filename)
    try:
        payload = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QwenArtifactManifestError(f"cannot load packaged artifact manifest: {filename}") from exc
    return decode_artifact_manifest(payload)


QWEN_RELEASE_ARTIFACT = _load_builtin_manifest("qwen3_asr_0_6b.json")
QWEN_REFERENCE_ARTIFACT = _load_builtin_manifest("qwen3_asr_1_7b.json")

_registry = {
    QWEN_RELEASE_ARTIFACT.key: QWEN_RELEASE_ARTIFACT,
    QWEN_REFERENCE_ARTIFACT.key: QWEN_REFERENCE_ARTIFACT,
}
if len(_registry) != 2:
    raise QwenArtifactManifestError("Qwen ASR artifact keys must be unique")
if {artifact.role for artifact in _registry.values()} != {
    ArtifactRole.RELEASE,
    ArtifactRole.REFERENCE,
}:
    raise QwenArtifactManifestError("Qwen ASR registry requires release and reference roles")
QWEN_ARTIFACT_REGISTRY: Mapping[str, QwenASRArtifactSpec] = MappingProxyType(_registry)


def get_qwen_artifact(
    key: str, *, expected_role: ArtifactRole | None = None
) -> QwenASRArtifactSpec:
    try:
        artifact = QWEN_ARTIFACT_REGISTRY[key]
    except KeyError as exc:
        raise QwenArtifactManifestError(f"unknown Qwen ASR artifact: {key}") from exc
    if expected_role is not None and artifact.role is not expected_role:
        raise QwenArtifactRoleError(
            f"artifact {key} has role {artifact.role.value}, expected {expected_role.value}"
        )
    return artifact


__all__ = [
    "ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "QWEN_ARTIFACT_REGISTRY",
    "QWEN_REFERENCE_ARTIFACT",
    "QWEN_RELEASE_ARTIFACT",
    "ArtifactFile",
    "ArtifactRole",
    "ArtifactSource",
    "QwenASRArtifactSpec",
    "QwenArtifactError",
    "QwenArtifactManifestError",
    "QwenArtifactPermissionError",
    "QwenArtifactPathError",
    "QwenArtifactRoleError",
    "QWEN_ASR_MODEL_ROOT_ENV",
    "QwenArtifactSchemaError",
    "canonical_manifest_json",
    "decode_artifact_manifest",
    "default_asr_model_root",
    "get_qwen_artifact",
    "validate_local_model_directory",
    "validate_private_receipt",
]
