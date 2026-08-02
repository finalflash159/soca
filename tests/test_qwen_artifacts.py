from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from soca.asr.qwen_artifacts import (
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    QWEN_ARTIFACT_REGISTRY,
    QWEN_REFERENCE_ARTIFACT,
    QWEN_RELEASE_ARTIFACT,
    ArtifactRole,
    QwenArtifactManifestError,
    QwenArtifactPermissionError,
    QwenArtifactRoleError,
    QwenArtifactSchemaError,
    QwenASRArtifactSpec,
    decode_artifact_manifest,
    default_asr_model_root,
    get_qwen_artifact,
    validate_private_receipt,
)


def test_registry_has_one_release_and_one_reference_artifact() -> None:
    assert set(QWEN_ARTIFACT_REGISTRY) == {
        "qwen3_asr_0_6b",
        "qwen3_asr_1_7b",
    }
    assert QWEN_RELEASE_ARTIFACT.role is ArtifactRole.RELEASE
    assert QWEN_REFERENCE_ARTIFACT.role is ArtifactRole.REFERENCE
    assert QWEN_RELEASE_ARTIFACT.upstream.revision == (
        "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
    )
    assert QWEN_REFERENCE_ARTIFACT.upstream.revision == (
        "7278e1e70fe206f11671096ffdd38061171dd6e5"
    )
    assert QWEN_RELEASE_ARTIFACT.mirror is None
    assert QWEN_REFERENCE_ARTIFACT.mirror is None


def test_artifact_file_manifests_pin_all_model_bytes() -> None:
    release_weights = QWEN_RELEASE_ARTIFACT.file("model.safetensors")
    assert release_weights.size == 1_876_091_704
    assert release_weights.sha256 == (
        "79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea"
    )

    reference_shards = [
        file for file in QWEN_REFERENCE_ARTIFACT.files if file.path.endswith(".safetensors")
    ]
    assert [(file.path, file.size, file.sha256) for file in reference_shards] == [
        (
            "model-00001-of-00002.safetensors",
            4_220_320_824,
            "a4cd1f1a04d90b757dc7f7dd26254e69a013b19e80efe590a83c6a3bde8608d6",
        ),
        (
            "model-00002-of-00002.safetensors",
            478_200_688,
            "6e0b9d9e09e2e0238e7ef3cc8a484ab387e91b90f1900bedf88bc92d7929ccfc",
        ),
    ]


def test_manifest_canonical_round_trip_has_stable_digest() -> None:
    encoded = QWEN_RELEASE_ARTIFACT.canonical_json
    assert encoded == QWEN_RELEASE_ARTIFACT.canonical_json
    assert " " not in encoded

    decoded = decode_artifact_manifest(json.loads(encoded))

    assert decoded == QWEN_RELEASE_ARTIFACT
    assert decoded.digest == QWEN_RELEASE_ARTIFACT.digest
    assert len(decoded.digest) == 64


def test_unknown_manifest_schema_is_rejected() -> None:
    payload = QWEN_RELEASE_ARTIFACT.to_manifest_dict()
    payload["schema_version"] = ARTIFACT_MANIFEST_SCHEMA_VERSION + 1

    with pytest.raises(QwenArtifactSchemaError, match="unsupported"):
        decode_artifact_manifest(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("key", "../release"),
        ("revision", "main"),
        ("file_path", "../model.safetensors"),
        ("file_path", "/tmp/model.safetensors"),
        ("file_path", "weights\\model.safetensors"),
    ],
)
def test_mutable_or_unsafe_manifest_identity_is_rejected(field: str, value: str) -> None:
    payload = QWEN_RELEASE_ARTIFACT.to_manifest_dict()
    if field == "revision":
        payload["upstream"]["revision"] = value  # type: ignore[index]
    elif field == "file_path":
        payload["files"][0]["path"] = value  # type: ignore[index]
    else:
        payload[field] = value

    with pytest.raises(QwenArtifactManifestError):
        decode_artifact_manifest(payload)


def test_registry_enforces_expected_role() -> None:
    with pytest.raises(QwenArtifactRoleError, match="release"):
        get_qwen_artifact("qwen3_asr_1_7b", expected_role=ArtifactRole.RELEASE)


def test_direct_spec_construction_rejects_untyped_role() -> None:
    fields = QWEN_RELEASE_ARTIFACT.to_manifest_dict()
    fields.pop("schema_version")

    with pytest.raises(QwenArtifactManifestError, match="role"):
        QwenASRArtifactSpec(**fields)  # type: ignore[arg-type]


def test_boolean_schema_version_is_rejected() -> None:
    payload = QWEN_RELEASE_ARTIFACT.to_manifest_dict()
    payload["schema_version"] = True

    with pytest.raises(QwenArtifactSchemaError, match="unsupported"):
        decode_artifact_manifest(payload)


def test_default_model_root_uses_absolute_xdg_data_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    assert default_asr_model_root() == tmp_path / "soca" / "models" / "asr"
    assert QWEN_RELEASE_ARTIFACT.model_path() == (
        tmp_path
        / "soca"
        / "models"
        / "asr"
        / "qwen3_asr_0_6b"
        / QWEN_RELEASE_ARTIFACT.upstream.revision
    )


def test_default_model_root_rejects_relative_xdg_data_home(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", "relative/data")

    with pytest.raises(QwenArtifactManifestError, match="absolute"):
        default_asr_model_root()


def test_receipt_must_be_private_regular_file(tmp_path: Path) -> None:
    receipt = tmp_path / "release.json"
    receipt.write_text("{}", encoding="utf-8")
    receipt.chmod(0o600)

    validate_private_receipt(receipt)

    receipt.chmod(0o644)
    with pytest.raises(QwenArtifactPermissionError, match="private"):
        validate_private_receipt(receipt)


def test_receipt_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    target.chmod(0o600)
    receipt = tmp_path / "receipt.json"
    receipt.symlink_to(target)

    with pytest.raises(QwenArtifactPermissionError, match="symlink"):
        validate_private_receipt(receipt)


def test_receipt_with_symlinked_parent_is_rejected(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    receipt = real_root / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    receipt.chmod(0o600)
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(QwenArtifactPermissionError, match="symlink"):
        validate_private_receipt(linked_root / "receipt.json")


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are not portable")
def test_model_path_rejects_non_absolute_override(tmp_path: Path) -> None:
    with pytest.raises(QwenArtifactManifestError, match="absolute"):
        QWEN_RELEASE_ARTIFACT.model_path(Path(tmp_path.name))


def test_model_path_rejects_parent_traversal(tmp_path: Path) -> None:
    unsafe_root = tmp_path / "staging" / ".." / "models"

    with pytest.raises(QwenArtifactManifestError, match="traversal"):
        QWEN_RELEASE_ARTIFACT.model_path(unsafe_root)


def test_qwen_runtime_defaults_share_the_release_artifact() -> None:
    from soca.asr.qwen_backend import DEFAULT_QWEN_MODEL_ID as backend_default
    from soca.asr.qwen_service_client import DEFAULT_QWEN_MODEL_ID as client_default
    from soca.asr.qwen_service_server import DEFAULT_QWEN_MODEL_ID as server_default

    expected = QWEN_RELEASE_ARTIFACT.upstream.repo_id
    assert backend_default == expected
    assert client_default == expected
    assert server_default == expected


def test_registry_import_does_not_load_qwen_or_open_network() -> None:
    script = """
import sys

def reject_network(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo'}:
        raise RuntimeError(f'network attempted during artifact import: {event}')

sys.addaudithook(reject_network)
import soca.asr.qwen_artifacts as artifacts
assert artifacts.QWEN_RELEASE_ARTIFACT.key == 'qwen3_asr_0_6b'
assert not any(name == 'qwen_asr' or name.startswith('qwen_asr.') for name in sys.modules)
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
