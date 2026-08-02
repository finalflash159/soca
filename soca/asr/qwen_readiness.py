from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .qwen_artifacts import (
    QwenArtifactError,
    QwenASRArtifactSpec,
    default_asr_model_root,
    validate_private_receipt,
)
from .qwen_store import ArtifactState, QwenArtifactStore

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_ROOT = REPO_ROOT / "runtime" / "qwen-asr"


class QwenReadinessState(StrEnum):
    MISSING = "missing"
    INVALID = "invalid"
    PROVISIONED = "provisioned"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class QwenStaticReadiness:
    state: QwenReadinessState
    artifact_key: str
    detail: str
    revision: str
    artifact_digest: str
    runtime_lock_digest: str | None
    no_fallback_attempted: bool = True


def inspect_qwen_readiness(
    spec: QwenASRArtifactSpec,
    *,
    store_root: Path | None = None,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    system: str | None = None,
    machine: str | None = None,
) -> QwenStaticReadiness:
    actual_system = system or platform.system()
    actual_machine = machine or platform.machine()
    if (actual_system, actual_machine) != ("Darwin", "arm64"):
        return _result(
            spec,
            QwenReadinessState.UNSUPPORTED,
            f"unsupported platform {actual_system}/{actual_machine}; no fallback attempted",
        )
    runtime_error = _inspect_runtime(spec, runtime_root)
    if runtime_error is not None:
        return _result(spec, QwenReadinessState.INVALID, runtime_error)
    try:
        inspection = QwenArtifactStore(store_root or default_asr_model_root()).inspect(spec)
    except (OSError, RuntimeError, ValueError) as exc:
        return _result(spec, QwenReadinessState.INVALID, f"artifact inspection failed: {exc}")
    if inspection.state is ArtifactState.MISSING:
        return _result(
            spec,
            QwenReadinessState.MISSING,
            "artifact missing; run scripts/provision_qwen_asr.py install",
        )
    if inspection.state is ArtifactState.INVALID:
        return _result(spec, QwenReadinessState.INVALID, inspection.detail)
    return _result(
        spec,
        QwenReadinessState.PROVISIONED,
        (
            f"{spec.key} · {spec.upstream.revision[:7]} · {spec.device}/{spec.dtype} · "
            "service stopped · artifact verified"
        ),
    )


def _inspect_runtime(spec: QwenASRArtifactSpec, root: Path) -> str | None:
    lock_path = root / "uv.lock"
    receipt_path = root / ".runtime-receipt.json"
    python_path = root / ".venv" / "bin" / "python"
    try:
        validate_private_receipt(receipt_path)
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        lock_digest = _sha256(lock_path)
    except QwenArtifactError as exc:
        return str(exc)
    except (OSError, json.JSONDecodeError) as exc:
        return f"worker runtime missing or unreadable: {type(exc).__name__}"
    if not isinstance(payload, Mapping):
        return "worker runtime receipt is invalid or not private"
    expected_fields = {
        "schema_version",
        "python",
        "uv",
        "lock_sha256",
        "soca_wheel_sha256",
        "environment",
    }
    if set(payload) != expected_fields:
        return "worker runtime receipt fields do not match schema"
    if (
        payload.get("schema_version") != 1
        or payload.get("lock_sha256") != lock_digest
        or spec.runtime_lock_digest != lock_digest
        or payload.get("environment") != str((root / ".venv").resolve())
    ):
        return "worker runtime identity does not match artifact"
    if not python_path.is_file():
        return "worker Python executable is missing"
    return None


def _result(
    spec: QwenASRArtifactSpec,
    state: QwenReadinessState,
    detail: str,
) -> QwenStaticReadiness:
    return QwenStaticReadiness(
        state=state,
        artifact_key=spec.key,
        detail=detail,
        revision=spec.upstream.revision,
        artifact_digest=spec.digest,
        runtime_lock_digest=spec.runtime_lock_digest,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "QwenReadinessState",
    "QwenStaticReadiness",
    "inspect_qwen_readiness",
]
