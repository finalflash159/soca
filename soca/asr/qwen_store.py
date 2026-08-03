from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import math
import os
import platform
import shutil
import stat
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from safetensors import safe_open

from .qwen_artifacts import ArtifactSource, QwenASRArtifactSpec

RECEIPT_SCHEMA_VERSION = 1


class QwenStoreError(RuntimeError):
    pass


class ArtifactInvalid(QwenStoreError):
    pass


class MirrorNotPinned(QwenStoreError):
    pass


class InsufficientArtifactDisk(QwenStoreError):
    pass


class ProvisionLockBusy(QwenStoreError):
    pass


class UnsupportedArtifactPlatform(QwenStoreError):
    pass


class WorkerRuntimeInvalid(QwenStoreError):
    pass


class ArtifactSourceKind(StrEnum):
    MIRROR = "mirror"
    UPSTREAM = "upstream"


class ArtifactState(StrEnum):
    MISSING = "missing"
    INVALID = "invalid"
    PROVISIONED = "provisioned"


@dataclass(frozen=True, slots=True)
class ArtifactInspection:
    artifact_key: str
    state: ArtifactState
    model_path: Path
    detail: str


@dataclass(frozen=True, slots=True)
class ArtifactPreflight:
    artifact_key: str
    platform: str
    architecture: str
    final_bytes: int
    staging_bytes: int
    reusable_bytes: int
    required_free_bytes: int
    free_bytes: int
    runtime_lock_digest: str


@dataclass(frozen=True, slots=True)
class ReceiptFile:
    path: str
    size: int
    sha256: str
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size": self.size,
            "sha256": self.sha256,
            "device": self.device,
            "inode": self.inode,
            "mtime_ns": self.mtime_ns,
            "ctime_ns": self.ctime_ns,
        }


@dataclass(frozen=True, slots=True)
class ArtifactReceipt:
    artifact_key: str
    artifact_role: str
    artifact_digest: str
    source_kind: ArtifactSourceKind
    source: ArtifactSource
    model_path: str
    runtime_lock_digest: str
    installed_at: str
    files: tuple[ReceiptFile, ...]
    health: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "artifact_key": self.artifact_key,
            "artifact_role": self.artifact_role,
            "artifact_digest": self.artifact_digest,
            "source_kind": self.source_kind.value,
            "source": self.source.to_dict(),
            "model_path": self.model_path,
            "runtime_lock_digest": self.runtime_lock_digest,
            "installed_at": self.installed_at,
            "files": [entry.to_dict() for entry in self.files],
            "health": dict(self.health),
        }


HealthProbe = Callable[[Path], Mapping[str, object]]
ProgressCallback = Callable[[str, int, int], None]
SnapshotFetch = Callable[..., str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _private_directory(path: Path) -> None:
    try:
        _reject_symlink_components(path)
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        _reject_symlink_components(path)
        metadata = path.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise ArtifactInvalid(f"artifact directory is not a private directory: {path}")
        _chmod_no_follow(path, 0o700)
    except OSError as exc:
        raise ArtifactInvalid(f"artifact directory cannot be made private: {path}") from exc


def _reject_symlink_components(path: Path) -> None:
    for component in (path, *path.parents):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactInvalid(f"artifact path contains a symlink: {component}")


def _chmod_no_follow(path: Path, mode: int) -> None:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        raise ArtifactInvalid(f"artifact tree contains a symlink: {path}")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if stat.S_ISDIR(metadata.st_mode):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_file(
    source: Path,
    destination: Path,
    *,
    progress: ProgressCallback | None,
    relative_path: str,
) -> None:
    cloned = False
    if platform.system() == "Darwin":
        clonefile = ctypes.CDLL(None, use_errno=True).clonefile
        clonefile.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)
        clonefile.restype = ctypes.c_int
        if clonefile(os.fsencode(source), os.fsencode(destination), 0) == 0:
            cloned = True
        else:
            error = ctypes.get_errno()
            if error not in {errno.EXDEV, errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
                raise OSError(error, os.strerror(error), str(source))
    if not cloned:
        copied = 0
        total = source.stat().st_size
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            while block := input_stream.read(1024 * 1024):
                output_stream.write(block)
                copied += len(block)
                if progress is not None:
                    progress(relative_path, copied, total)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    else:
        with destination.open("rb") as output_stream:
            os.fsync(output_stream.fileno())
        if progress is not None:
            size = source.stat().st_size
            progress(relative_path, size, size)
    os.chmod(destination, 0o600)


def _validate_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactInvalid(f"{label} is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise ArtifactInvalid(f"{label} must be a JSON object")
    return payload


def _validate_model_structure(root: Path, spec: QwenASRArtifactSpec) -> None:
    config = _validate_json(root / "config.json", "config.json")
    if config.get("model_type") != "qwen3_asr":
        raise ArtifactInvalid("config.json has the wrong model_type")
    _validate_json(root / "tokenizer_config.json", "tokenizer_config.json")
    _validate_json(root / "preprocessor_config.json", "preprocessor_config.json")
    weights = tuple(entry.path for entry in spec.files if entry.path.endswith(".safetensors"))
    if not weights:
        raise ArtifactInvalid("artifact has no safetensors weights")
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        index = _validate_json(index_path, "model.safetensors.index.json")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ArtifactInvalid("safetensors index has no weight_map")
        referenced = {value for value in weight_map.values() if isinstance(value, str)}
        if referenced != set(weights):
            raise ArtifactInvalid("safetensors index does not reference the expected shards")
    elif len(weights) != 1 or weights[0] != "model.safetensors":
        raise ArtifactInvalid("sharded safetensors artifact is missing its index")
    for relative in weights:
        try:
            with safe_open(root / relative, framework="numpy") as handle:
                if not list(handle.keys()):
                    raise ArtifactInvalid(f"safetensors file has no tensors: {relative}")
        except ArtifactInvalid:
            raise
        except Exception as exc:
            raise ArtifactInvalid(f"safetensors header is invalid: {relative}") from exc


def _sanitize_health(payload: Mapping[str, object]) -> Mapping[str, object]:
    for key, value in payload.items():
        if not isinstance(key, str):
            raise ArtifactInvalid("offline health probe returned a non-string field")
        if any(marker in key.casefold() for marker in ("token", "secret", "credential")):
            raise ArtifactInvalid("offline health probe returned a sensitive field")
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise ArtifactInvalid("offline health probe returned a non-scalar field")
        if isinstance(value, float) and not math.isfinite(value):
            raise ArtifactInvalid("offline health probe returned a non-finite metric")
    sanitized = {key: value for key, value in payload.items() if key != "transcript"}
    transcript = payload.get("transcript")
    if not isinstance(transcript, str) or not transcript.strip():
        raise ArtifactInvalid("offline health probe returned an empty transcript")
    sanitized["transcript_nonempty"] = True
    sanitized["transcript_sha256"] = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    return sanitized


class QwenSnapshotResolver:
    def __init__(self, fetch: SnapshotFetch | None = None) -> None:
        if fetch is None:
            from huggingface_hub import snapshot_download

            fetch = snapshot_download
        self._fetch = fetch

    def resolve(
        self,
        spec: QwenASRArtifactSpec,
        *,
        source_kind: ArtifactSourceKind,
        cache_only: bool,
        token: str | None,
    ) -> Path:
        source = spec.mirror if source_kind is ArtifactSourceKind.MIRROR else spec.upstream
        if source is None:
            raise MirrorNotPinned("artifact mirror is not pinned")
        try:
            resolved = self._fetch(
                repo_id=source.repo_id,
                revision=source.revision,
                allow_patterns=[entry.path for entry in spec.files],
                local_files_only=cache_only,
                token=token,
                etag_timeout=10.0,
                max_workers=4,
            )
        except Exception as exc:
            raise ArtifactInvalid(
                f"could not resolve exact {source_kind.value} snapshot for {spec.key}"
            ) from exc
        path = Path(resolved)
        if not path.is_absolute() or not path.is_dir():
            raise ArtifactInvalid("snapshot resolver returned an invalid local directory")
        return path


class QwenArtifactStore:
    def __init__(
        self,
        root: Path,
        *,
        disk_free: Callable[[Path], int] | None = None,
    ) -> None:
        expanded = root.expanduser()
        if not expanded.is_absolute() or ".." in expanded.parts:
            raise ValueError("Qwen artifact store root must be absolute without traversal")
        try:
            _reject_symlink_components(expanded)
        except OSError as exc:
            raise ArtifactInvalid("Qwen artifact store path cannot be inspected") from exc
        self.root = expanded
        self._disk_free = disk_free or (lambda path: shutil.disk_usage(path).free)

    def preflight(
        self,
        spec: QwenASRArtifactSpec,
        source_root: Path,
        *,
        runtime_lock: Path,
        system: str | None = None,
        machine: str | None = None,
    ) -> ArtifactPreflight:
        actual_system = system or platform.system()
        actual_machine = machine or platform.machine()
        if (actual_system, actual_machine) != ("Darwin", "arm64"):
            raise UnsupportedArtifactPlatform(
                f"Qwen ASR artifacts support Darwin/arm64, got "
                f"{actual_system}/{actual_machine}"
            )
        if spec.runtime_lock_digest is None:
            raise WorkerRuntimeInvalid("artifact does not pin a worker runtime lock")
        try:
            lock_digest = _sha256(runtime_lock)
        except OSError as exc:
            raise WorkerRuntimeInvalid("Qwen worker runtime lock is missing") from exc
        if lock_digest != spec.runtime_lock_digest:
            raise WorkerRuntimeInvalid("Qwen worker runtime lock digest does not match")
        try:
            if not source_root.is_dir() or source_root.is_symlink():
                raise ArtifactInvalid("artifact source must be a local directory")
            _private_directory(self.root)
        except OSError as exc:
            raise ArtifactInvalid("artifact source or target store is unavailable") from exc
        reusable = (
            spec.total_bytes
            if self.inspect(spec).state is ArtifactState.PROVISIONED
            else 0
        )
        staging = 0 if reusable else spec.total_bytes
        required = staging
        free = self._disk_free(self.root)
        if free < required:
            raise InsufficientArtifactDisk(
                f"artifact requires {required} free bytes, found {free}"
            )
        return ArtifactPreflight(
            artifact_key=spec.key,
            platform=actual_system,
            architecture=actual_machine,
            final_bytes=spec.total_bytes,
            staging_bytes=staging,
            reusable_bytes=reusable,
            required_free_bytes=required,
            free_bytes=free,
            runtime_lock_digest=lock_digest,
        )

    @contextmanager
    def provision_lock(self) -> Iterator[None]:
        _private_directory(self.root)
        lock_path = self.root / ".provision.lock"
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ProvisionLockBusy("another Qwen artifact provisioner owns the store") from exc
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def install_from_snapshot(
        self,
        spec: QwenASRArtifactSpec,
        source_root: Path,
        *,
        source_kind: ArtifactSourceKind,
        health_probe: HealthProbe,
        runtime_lock: Path,
        system: str | None = None,
        machine: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> ArtifactReceipt:
        source = spec.mirror if source_kind is ArtifactSourceKind.MIRROR else spec.upstream
        if source is None:
            raise MirrorNotPinned("artifact mirror is not pinned")
        self.preflight(
            spec,
            source_root,
            runtime_lock=runtime_lock,
            system=system,
            machine=machine,
        )
        with self.provision_lock():
            existing = self._load_if_valid(spec)
            if existing is not None:
                return existing
            _private_directory(self.root / "receipts")
            target = spec.model_path(self.root)
            _private_directory(target.parent)
            if self._disk_free(self.root) < spec.total_bytes:
                raise InsufficientArtifactDisk(
                    f"artifact requires {spec.total_bytes} bytes before staging"
                )
            staging = self.root / f".staging-{spec.key}-{uuid4().hex}"
            _private_directory(staging)
            activated = False
            try:
                self._materialize(spec, source_root, staging, progress=progress)
                self.verify_directory(spec, staging, deep=True)
                health = _sanitize_health(health_probe(staging))
                if target.exists():
                    raise ArtifactInvalid("artifact generation exists without a valid receipt")
                os.replace(staging, target)
                activated = True
                try:
                    self._make_read_only(target)
                    _fsync_directory(target.parent)
                    receipt = self._build_receipt(spec, source_kind, source, target, health)
                    self._write_receipt(spec.receipt_path(self.root), receipt)
                except OSError as exc:
                    raise ArtifactInvalid("artifact activation could not be persisted") from exc
                return receipt
            except Exception:
                if activated and target.exists():
                    self._make_writable(target)
                    shutil.rmtree(target)
                    spec.receipt_path(self.root).unlink(missing_ok=True)
                    _fsync_directory(target.parent)
                raise
            finally:
                if staging.exists():
                    self._make_writable(staging)
                    shutil.rmtree(staging)

    def refresh_receipt(
        self,
        spec: QwenASRArtifactSpec,
        *,
        source_kind: ArtifactSourceKind,
        health_probe: HealthProbe,
        runtime_lock: Path,
    ) -> ArtifactReceipt:
        """Re-issue identity for unchanged local bytes after a spec change.

        A device or dtype change intentionally changes the manifest digest. This
        operation never downloads or replaces model files: it hashes the
        existing generation, runs the supplied offline health probe, and then
        writes a new private receipt only after both checks pass.
        """
        source = spec.mirror if source_kind is ArtifactSourceKind.MIRROR else spec.upstream
        if source is None:
            raise MirrorNotPinned("artifact mirror is not pinned")
        if spec.runtime_lock_digest is None:
            raise WorkerRuntimeInvalid("artifact does not pin a worker runtime lock")
        try:
            lock_digest = _sha256(runtime_lock)
        except OSError as exc:
            raise WorkerRuntimeInvalid("Qwen worker runtime lock is missing") from exc
        if lock_digest != spec.runtime_lock_digest:
            raise WorkerRuntimeInvalid("Qwen worker runtime lock digest does not match")

        target = spec.model_path(self.root)
        if not target.is_dir() or target.is_symlink():
            raise ArtifactInvalid("cannot refresh a missing or indirect artifact generation")

        with self.provision_lock():
            self.verify_directory(spec, target, deep=True)
            health = _sanitize_health(health_probe(target))
            _private_directory(self.root / "receipts")
            receipt = self._build_receipt(spec, source_kind, source, target, health)
            self._write_receipt(spec.receipt_path(self.root), receipt)
            return receipt

    def verify(
        self,
        spec: QwenASRArtifactSpec,
        *,
        deep: bool,
        health_probe: HealthProbe | None = None,
    ) -> ArtifactReceipt:
        receipt = self._load_if_valid(spec)
        if receipt is None:
            raise ArtifactInvalid("artifact is not provisioned")
        if deep:
            if health_probe is None:
                raise ArtifactInvalid("deep verification requires an offline health probe")
            target = spec.model_path(self.root)
            self.verify_directory(spec, target, deep=True)
            _sanitize_health(health_probe(target))
        return receipt

    def inspect(self, spec: QwenASRArtifactSpec) -> ArtifactInspection:
        target = spec.model_path(self.root)
        receipt = spec.receipt_path(self.root)
        if not target.exists() and not receipt.exists():
            return ArtifactInspection(spec.key, ArtifactState.MISSING, target, "not installed")
        try:
            self.verify(spec, deep=False)
        except (QwenStoreError, OSError, ValueError) as exc:
            return ArtifactInspection(spec.key, ArtifactState.INVALID, target, str(exc))
        return ArtifactInspection(spec.key, ArtifactState.PROVISIONED, target, "quick identity verified")

    def gc(
        self,
        specs: tuple[QwenASRArtifactSpec, ...],
        *,
        dry_run: bool = False,
        generation: str | None = None,
    ) -> tuple[Path, ...]:
        active = {spec.model_path(self.root) for spec in specs if spec.model_path(self.root).exists()}
        candidates: list[Path] = []
        for spec in specs:
            artifact_root = self.root / spec.key
            if not artifact_root.is_dir() or artifact_root.is_symlink():
                continue
            for path in artifact_root.iterdir():
                if (
                    path.is_dir()
                    and not path.is_symlink()
                    and len(path.name) == 40
                    and all(character in "0123456789abcdef" for character in path.name)
                    and path not in active
                ):
                    candidates.append(path)
        candidates.sort()
        if generation is None:
            if dry_run:
                return tuple(candidates)
            raise ArtifactInvalid("gc deletion requires an explicit generation")
        if len(generation) != 40 or any(character not in "0123456789abcdef" for character in generation):
            raise ArtifactInvalid("gc generation must be a full immutable revision")
        selected = tuple(path for path in candidates if path.name == generation)
        if any(path.name == generation for path in active):
            raise ArtifactInvalid("gc cannot remove an active generation")
        if not selected:
            raise ArtifactInvalid("gc generation was not found")
        if dry_run:
            return selected
        for path in selected:
            self._make_writable(path)
            shutil.rmtree(path)
            _fsync_directory(path.parent)
        return selected

    def _materialize(
        self,
        spec: QwenASRArtifactSpec,
        source_root: Path,
        staging: Path,
        *,
        progress: ProgressCallback | None,
    ) -> None:
        for expected in spec.files:
            source = source_root / expected.path
            try:
                resolved = source.resolve(strict=True)
                metadata = resolved.stat()
            except OSError as exc:
                raise ArtifactInvalid(f"artifact source file is missing: {expected.path}") from exc
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected.size:
                raise ArtifactInvalid(f"artifact source size mismatch: {expected.path}")
            if _sha256(resolved) != expected.sha256:
                raise ArtifactInvalid(f"artifact source sha256 mismatch: {expected.path}")
            destination = staging / expected.path
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            _copy_file(
                resolved,
                destination,
                progress=progress,
                relative_path=expected.path,
            )
        _fsync_directory(staging)

    def verify_directory(
        self,
        spec: QwenASRArtifactSpec,
        root: Path,
        *,
        deep: bool,
    ) -> None:
        if root.is_symlink():
            raise ArtifactInvalid("artifact generation must not be a symlink")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ArtifactInvalid(
                    f"artifact tree contains a symlink: {path.relative_to(root)}"
                )
        actual = tuple(
            path.relative_to(root).as_posix()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        )
        expected_paths = tuple(entry.path for entry in spec.files)
        if actual != expected_paths:
            raise ArtifactInvalid("artifact file set does not match its manifest")
        for expected in spec.files:
            path = root / expected.path
            metadata = path.lstat()
            if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                raise ArtifactInvalid(f"artifact file must be regular and non-symlink: {expected.path}")
            if metadata.st_size != expected.size:
                raise ArtifactInvalid(f"artifact size mismatch: {expected.path}")
            if deep and _sha256(path) != expected.sha256:
                raise ArtifactInvalid(f"artifact sha256 mismatch: {expected.path}")
        if deep:
            _validate_model_structure(root, spec)

    def _build_receipt(
        self,
        spec: QwenASRArtifactSpec,
        source_kind: ArtifactSourceKind,
        source: ArtifactSource,
        target: Path,
        health: Mapping[str, object],
    ) -> ArtifactReceipt:
        if spec.runtime_lock_digest is None:
            raise ArtifactInvalid("artifact does not pin a runtime lock")
        files = []
        for expected in spec.files:
            metadata = (target / expected.path).stat()
            files.append(
                ReceiptFile(
                    path=expected.path,
                    size=metadata.st_size,
                    sha256=expected.sha256,
                    device=metadata.st_dev,
                    inode=metadata.st_ino,
                    mtime_ns=metadata.st_mtime_ns,
                    ctime_ns=metadata.st_ctime_ns,
                )
            )
        return ArtifactReceipt(
            artifact_key=spec.key,
            artifact_role=spec.role.value,
            artifact_digest=spec.digest,
            source_kind=source_kind,
            source=source,
            model_path=str(target),
            runtime_lock_digest=spec.runtime_lock_digest,
            installed_at=datetime.now(UTC).isoformat(),
            files=tuple(files),
            health=health,
        )

    def _write_receipt(self, path: Path, receipt: ArtifactReceipt) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        encoded = (json.dumps(receipt.to_dict(), sort_keys=True, separators=(",", ":")) + "\n").encode()
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            os.chmod(path, 0o600)
            _fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def _load_if_valid(self, spec: QwenASRArtifactSpec) -> ArtifactReceipt | None:
        path = spec.receipt_path(self.root)
        target = spec.model_path(self.root)
        if not path.exists() and not target.exists():
            return None
        receipt = self.load_receipt(spec)
        self.verify_directory(spec, target, deep=False)
        for expected, recorded in zip(spec.files, receipt.files, strict=True):
            metadata = (target / expected.path).stat()
            identity = (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )
            if identity != (
                recorded.device,
                recorded.inode,
                recorded.size,
                recorded.mtime_ns,
                recorded.ctime_ns,
            ):
                raise ArtifactInvalid(f"artifact changed after activation: {expected.path}")
        return receipt

    def load_receipt(self, spec: QwenASRArtifactSpec) -> ArtifactReceipt:
        path = spec.receipt_path(self.root)
        try:
            metadata = path.lstat()
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactInvalid("artifact receipt is missing or unreadable") from exc
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ArtifactInvalid("artifact receipt is not a private regular file")
        if not isinstance(payload, Mapping):
            raise ArtifactInvalid("artifact receipt must be a JSON object")
        expected_keys = {
            "schema_version",
            "artifact_key",
            "artifact_role",
            "artifact_digest",
            "source_kind",
            "source",
            "model_path",
            "runtime_lock_digest",
            "installed_at",
            "files",
            "health",
        }
        if set(payload) != expected_keys:
            raise ArtifactInvalid("artifact receipt fields do not match the schema")
        try:
            if payload.get("schema_version") != RECEIPT_SCHEMA_VERSION:
                raise ArtifactInvalid("artifact receipt schema is unsupported")
            source_payload = payload["source"]
            if not isinstance(source_payload, Mapping) or set(source_payload) != {
                "repo_id",
                "revision",
            }:
                raise ArtifactInvalid("artifact receipt source fields are invalid")
            raw_files = payload["files"]
            if not isinstance(raw_files, list):
                raise ArtifactInvalid("artifact receipt files must be an array")
            file_keys = {"path", "size", "sha256", "device", "inode", "mtime_ns", "ctime_ns"}
            if any(not isinstance(item, Mapping) or set(item) != file_keys for item in raw_files):
                raise ArtifactInvalid("artifact receipt file fields are invalid")
            health = payload["health"]
            if not isinstance(health, Mapping) or "transcript" in health:
                raise ArtifactInvalid("artifact receipt health fields are invalid")
            receipt = ArtifactReceipt(
                artifact_key=payload["artifact_key"],
                artifact_role=payload["artifact_role"],
                artifact_digest=payload["artifact_digest"],
                source_kind=ArtifactSourceKind(payload["source_kind"]),
                source=ArtifactSource(source_payload["repo_id"], source_payload["revision"]),
                model_path=payload["model_path"],
                runtime_lock_digest=payload["runtime_lock_digest"],
                installed_at=payload["installed_at"],
                files=tuple(ReceiptFile(**item) for item in raw_files),
                health=health,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactInvalid("artifact receipt fields are invalid") from exc
        expected_source = spec.mirror if receipt.source_kind is ArtifactSourceKind.MIRROR else spec.upstream
        if (
            receipt.artifact_key != spec.key
            or receipt.artifact_role != spec.role.value
            or receipt.artifact_digest != spec.digest
            or receipt.model_path != str(spec.model_path(self.root))
            or receipt.runtime_lock_digest != spec.runtime_lock_digest
            or tuple(item.path for item in receipt.files) != tuple(item.path for item in spec.files)
            or tuple(item.sha256 for item in receipt.files) != tuple(item.sha256 for item in spec.files)
            or receipt.source != expected_source
        ):
            raise ArtifactInvalid("artifact receipt identity does not match the selected spec")
        return receipt

    @staticmethod
    def _make_read_only(root: Path) -> None:
        for path in root.rglob("*"):
            _chmod_no_follow(path, 0o500 if path.is_dir() else 0o400)
        _chmod_no_follow(root, 0o500)

    @staticmethod
    def _make_writable(root: Path) -> None:
        _chmod_no_follow(root, 0o700)
        for path in root.rglob("*"):
            _chmod_no_follow(path, 0o700 if path.is_dir() else 0o600)


__all__ = [
    "ArtifactInvalid",
    "ArtifactInspection",
    "ArtifactPreflight",
    "ArtifactReceipt",
    "ArtifactSourceKind",
    "ArtifactState",
    "InsufficientArtifactDisk",
    "MirrorNotPinned",
    "ProvisionLockBusy",
    "QwenArtifactStore",
    "QwenSnapshotResolver",
    "QwenStoreError",
    "UnsupportedArtifactPlatform",
    "WorkerRuntimeInvalid",
]
