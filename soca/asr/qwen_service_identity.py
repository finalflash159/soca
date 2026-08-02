from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .qwen_artifacts import (
    QwenArtifactPathError,
    QwenASRArtifactSpec,
    validate_local_model_directory,
)

if TYPE_CHECKING:
    from .qwen_store import ArtifactReceipt

QWEN_SERVICE_PROTOCOL_VERSION = 2
REQUIRED_PACKAGE_VERSIONS = frozenset({"qwen-asr", "soca", "torch", "transformers"})


class QwenServiceIdentityError(RuntimeError):
    pass


class QwenServiceState(StrEnum):
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    FAILED = "failed"
    STOPPING = "stopping"
    STOPPED = "stopped"


class QwenLaunchMode(StrEnum):
    PROVISIONING = "provisioning"
    ACTIVE = "active"


@dataclass(frozen=True, slots=True)
class QwenServiceLaunch:
    spec: QwenASRArtifactSpec
    model_path: Path
    mode: QwenLaunchMode

    @classmethod
    def for_provisioning(cls, spec: QwenASRArtifactSpec, model_path: Path) -> QwenServiceLaunch:
        return cls(spec=spec, model_path=model_path, mode=QwenLaunchMode.PROVISIONING)

    @classmethod
    def for_active(cls, spec: QwenASRArtifactSpec, receipt: ArtifactReceipt) -> QwenServiceLaunch:
        expected_source = spec.mirror if receipt.source_kind.value == "mirror" else spec.upstream
        model_path = Path(receipt.model_path)
        if (
            receipt.artifact_key != spec.key
            or receipt.artifact_role != spec.role.value
            or receipt.artifact_digest != spec.digest
            or receipt.runtime_lock_digest != spec.runtime_lock_digest
            or receipt.source != expected_source
            or model_path.name != spec.upstream.revision
            or model_path.parent.name != spec.key
        ):
            raise QwenServiceIdentityError("artifact receipt does not match launch spec")
        return cls(
            spec=spec,
            model_path=model_path,
            mode=QwenLaunchMode.ACTIVE,
        )

    def __post_init__(self) -> None:
        if QWEN_SERVICE_PROTOCOL_VERSION < self.spec.minimum_protocol_version:
            raise QwenServiceIdentityError(
                "Qwen service protocol does not satisfy the artifact minimum"
            )
        try:
            path = validate_local_model_directory(self.model_path.expanduser())
        except QwenArtifactPathError as exc:
            raise QwenServiceIdentityError(str(exc)) from exc
        object.__setattr__(self, "model_path", path)


@dataclass(frozen=True, slots=True)
class QwenServiceIdentity:
    protocol_version: int
    state: QwenServiceState
    launch_mode: QwenLaunchMode
    artifact_key: str
    artifact_role: str
    upstream_revision: str
    mirror_revision: str | None
    artifact_digest: str
    runtime_lock_digest: str
    context_policy_digest: str | None
    backend: str
    device: str
    dtype: str
    package_versions: Mapping[str, str]
    pid: int
    uptime_ms: float
    in_flight: int
    supports_avg_logprob: bool
    last_failure_type: str | None
    no_fallback_attempted: bool

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> QwenServiceIdentity:
        expected = {
            "protocol_version",
            "state",
            "launch_mode",
            "artifact_key",
            "artifact_role",
            "upstream_revision",
            "mirror_revision",
            "artifact_digest",
            "runtime_lock_digest",
            "context_policy_digest",
            "backend",
            "device",
            "dtype",
            "package_versions",
            "pid",
            "uptime_ms",
            "in_flight",
            "supports_avg_logprob",
            "last_failure_type",
            "no_fallback_attempted",
        }
        if set(payload) != expected:
            raise QwenServiceIdentityError("service identity fields do not match protocol")
        packages = payload.get("package_versions")
        if not isinstance(packages, Mapping) or set(packages) != REQUIRED_PACKAGE_VERSIONS:
            raise QwenServiceIdentityError("service package identity is incomplete")
        if any(
            not isinstance(key, str) or not isinstance(value, str) or not value
            for key, value in packages.items()
        ):
            raise QwenServiceIdentityError("service package versions must be non-empty strings")
        try:
            identity = cls(
                protocol_version=_integer(payload, "protocol_version"),
                state=QwenServiceState(_string(payload, "state")),
                launch_mode=QwenLaunchMode(_string(payload, "launch_mode")),
                artifact_key=_string(payload, "artifact_key"),
                artifact_role=_string(payload, "artifact_role"),
                upstream_revision=_string(payload, "upstream_revision"),
                mirror_revision=_optional_string(payload, "mirror_revision"),
                artifact_digest=_string(payload, "artifact_digest"),
                runtime_lock_digest=_string(payload, "runtime_lock_digest"),
                context_policy_digest=_optional_string(payload, "context_policy_digest"),
                backend=_string(payload, "backend"),
                device=_string(payload, "device"),
                dtype=_string(payload, "dtype"),
                package_versions=dict(packages),
                pid=_integer(payload, "pid"),
                uptime_ms=_number(payload, "uptime_ms"),
                in_flight=_integer(payload, "in_flight"),
                supports_avg_logprob=_boolean(payload, "supports_avg_logprob"),
                last_failure_type=_optional_string(payload, "last_failure_type"),
                no_fallback_attempted=_boolean(payload, "no_fallback_attempted"),
            )
        except ValueError as exc:
            raise QwenServiceIdentityError("service identity enum is invalid") from exc
        if identity.protocol_version != QWEN_SERVICE_PROTOCOL_VERSION:
            raise QwenServiceIdentityError("Qwen service protocol version is incompatible")
        if identity.pid < 1 or identity.uptime_ms < 0 or identity.in_flight < 0:
            raise QwenServiceIdentityError("service lifecycle metrics are invalid")
        return identity

    def assert_matches(self, launch: QwenServiceLaunch) -> None:
        spec = launch.spec
        expected = (
            launch.mode,
            spec.key,
            spec.role.value,
            spec.upstream.revision,
            spec.mirror.revision if spec.mirror is not None else None,
            spec.digest,
            spec.runtime_lock_digest,
            spec.context_policy_digest,
            spec.device,
            spec.dtype,
        )
        actual = (
            self.launch_mode,
            self.artifact_key,
            self.artifact_role,
            self.upstream_revision,
            self.mirror_revision,
            self.artifact_digest,
            self.runtime_lock_digest,
            self.context_policy_digest,
            self.device,
            self.dtype,
        )
        if actual != expected:
            raise QwenServiceIdentityError("Qwen service identity does not match launch")
        if self.backend != "qwen3_asr" or not self.no_fallback_attempted:
            raise QwenServiceIdentityError("Qwen service backend policy is invalid")

    def to_wire(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "state": self.state.value,
            "launch_mode": self.launch_mode.value,
            "artifact_key": self.artifact_key,
            "artifact_role": self.artifact_role,
            "upstream_revision": self.upstream_revision,
            "mirror_revision": self.mirror_revision,
            "artifact_digest": self.artifact_digest,
            "runtime_lock_digest": self.runtime_lock_digest,
            "context_policy_digest": self.context_policy_digest,
            "backend": self.backend,
            "device": self.device,
            "dtype": self.dtype,
            "package_versions": dict(self.package_versions),
            "pid": self.pid,
            "uptime_ms": self.uptime_ms,
            "in_flight": self.in_flight,
            "supports_avg_logprob": self.supports_avg_logprob,
            "last_failure_type": self.last_failure_type,
            "no_fallback_attempted": self.no_fallback_attempted,
        }


def _string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise QwenServiceIdentityError(f"{key} must be a non-empty string")
    return value


def _optional_string(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is not None and not isinstance(value, str):
        raise QwenServiceIdentityError(f"{key} must be a string or null")
    return value


def _integer(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise QwenServiceIdentityError(f"{key} must be an integer")
    return value


def _number(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QwenServiceIdentityError(f"{key} must be numeric")
    return float(value)


def _boolean(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise QwenServiceIdentityError(f"{key} must be a boolean")
    return value


__all__ = [
    "QWEN_SERVICE_PROTOCOL_VERSION",
    "QwenLaunchMode",
    "QwenServiceIdentity",
    "QwenServiceIdentityError",
    "QwenServiceLaunch",
    "QwenServiceState",
]
