from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from soca.asr.qwen_artifacts import QWEN_RELEASE_ARTIFACT
from soca.asr.qwen_service_identity import (
    QWEN_SERVICE_PROTOCOL_VERSION,
    QwenLaunchMode,
    QwenServiceIdentity,
    QwenServiceIdentityError,
    QwenServiceLaunch,
    QwenServiceState,
)
from soca.asr.qwen_store import ArtifactReceipt, ArtifactSourceKind


def _identity(launch: QwenServiceLaunch) -> QwenServiceIdentity:
    spec = launch.spec
    return QwenServiceIdentity(
        protocol_version=QWEN_SERVICE_PROTOCOL_VERSION,
        state=QwenServiceState.READY,
        launch_mode=launch.mode,
        artifact_key=spec.key,
        artifact_role=spec.role.value,
        upstream_revision=spec.upstream.revision,
        mirror_revision=spec.mirror.revision if spec.mirror is not None else None,
        artifact_digest=spec.digest,
        runtime_lock_digest=spec.runtime_lock_digest or "",
        context_policy_digest=spec.context_policy_digest,
        backend="qwen3_asr",
        device=spec.device,
        dtype=spec.dtype,
        package_versions={
            "qwen-asr": "0.0.6",
            "soca": "0.1.0",
            "torch": "2.13.0",
            "transformers": "4.57.6",
        },
        pid=123,
        uptime_ms=12.5,
        in_flight=0,
        supports_avg_logprob=True,
        last_failure_type=None,
        no_fallback_attempted=True,
    )


def test_identity_round_trip_and_launch_match(tmp_path: Path) -> None:
    model_path = tmp_path / QWEN_RELEASE_ARTIFACT.key / QWEN_RELEASE_ARTIFACT.upstream.revision
    model_path.mkdir(parents=True)
    launch = QwenServiceLaunch.for_provisioning(QWEN_RELEASE_ARTIFACT, model_path)

    decoded = QwenServiceIdentity.from_wire(_identity(launch).to_wire())
    decoded.assert_matches(launch)

    assert decoded.state is QwenServiceState.READY
    assert decoded.launch_mode is QwenLaunchMode.PROVISIONING
    assert decoded.no_fallback_attempted is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("protocol_version", QWEN_SERVICE_PROTOCOL_VERSION + 1, "protocol version"),
        ("artifact_key", "wrong-model", "does not match"),
        ("upstream_revision", "wrong-revision", "does not match"),
        ("artifact_digest", "wrong-digest", "does not match"),
        ("context_policy_digest", "wrong-context", "does not match"),
    ],
)
def test_identity_rejects_incompatible_launch_identity(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    launch = QwenServiceLaunch.for_provisioning(QWEN_RELEASE_ARTIFACT, model_path)
    payload = _identity(launch).to_wire()
    payload[field] = value

    with pytest.raises(QwenServiceIdentityError, match=message):
        QwenServiceIdentity.from_wire(payload).assert_matches(launch)


def test_identity_rejects_unknown_fields(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    launch = QwenServiceLaunch.for_provisioning(QWEN_RELEASE_ARTIFACT, model_path)
    payload = _identity(launch).to_wire()
    payload["unexpected"] = True

    with pytest.raises(QwenServiceIdentityError, match="fields"):
        QwenServiceIdentity.from_wire(payload)


def test_launch_rejects_artifact_requiring_newer_protocol(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    incompatible = replace(
        QWEN_RELEASE_ARTIFACT,
        minimum_protocol_version=QWEN_SERVICE_PROTOCOL_VERSION + 1,
    )

    with pytest.raises(QwenServiceIdentityError, match="artifact minimum"):
        QwenServiceLaunch.for_provisioning(incompatible, model_path)


def test_launch_rejects_model_path_with_symlink_ancestor(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    (real_parent / "model").mkdir(parents=True)
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(QwenServiceIdentityError, match="contains a symlink"):
        QwenServiceLaunch.for_provisioning(
            QWEN_RELEASE_ARTIFACT, linked_parent / "model"
        )


def test_active_launch_requires_matching_receipt_and_absolute_local_path(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / QWEN_RELEASE_ARTIFACT.key / QWEN_RELEASE_ARTIFACT.upstream.revision
    model_path.mkdir(parents=True)
    receipt = ArtifactReceipt(
        artifact_key=QWEN_RELEASE_ARTIFACT.key,
        artifact_role=QWEN_RELEASE_ARTIFACT.role.value,
        artifact_digest=QWEN_RELEASE_ARTIFACT.digest,
        source_kind=ArtifactSourceKind.UPSTREAM,
        source=QWEN_RELEASE_ARTIFACT.upstream,
        model_path=str(model_path),
        runtime_lock_digest=QWEN_RELEASE_ARTIFACT.runtime_lock_digest or "",
        installed_at="2026-08-02T00:00:00+00:00",
        files=(),
        health={},
    )

    launch = QwenServiceLaunch.for_active(QWEN_RELEASE_ARTIFACT, receipt)
    assert launch.mode is QwenLaunchMode.ACTIVE

    with pytest.raises(QwenServiceIdentityError, match="receipt"):
        QwenServiceLaunch.for_active(
            QWEN_RELEASE_ARTIFACT,
            replace(receipt, artifact_digest="wrong"),
        )
    with pytest.raises(QwenServiceIdentityError, match="receipt"):
        QwenServiceLaunch.for_active(
            QWEN_RELEASE_ARTIFACT,
            replace(receipt, model_path=str(tmp_path / "wrong")),
        )
    with pytest.raises(QwenServiceIdentityError, match="absolute"):
        QwenServiceLaunch.for_provisioning(QWEN_RELEASE_ARTIFACT, Path("relative"))
