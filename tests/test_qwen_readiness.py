from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from soca.asr.qwen_artifacts import QWEN_RELEASE_ARTIFACT
from soca.asr.qwen_readiness import QwenReadinessState, inspect_qwen_readiness
from soca.asr.qwen_store import ArtifactInspection, ArtifactState


def _runtime(tmp_path: Path):
    root = tmp_path / "runtime"
    python = root / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    lock = root / "uv.lock"
    lock.write_text("locked", encoding="utf-8")
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    receipt = root / ".runtime-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "python": "3.11.14",
                "uv": "0.11.16",
                "lock_sha256": digest,
                "soca_wheel_sha256": "a" * 64,
                "environment": str(root / ".venv"),
            }
        ),
        encoding="utf-8",
    )
    receipt.chmod(0o600)
    return root, replace(QWEN_RELEASE_ARTIFACT, runtime_lock_digest=digest)


def test_static_readiness_reports_unsupported_without_loading_model(tmp_path: Path) -> None:
    before = set(sys.modules)

    result = inspect_qwen_readiness(
        QWEN_RELEASE_ARTIFACT,
        store_root=tmp_path / "store",
        runtime_root=tmp_path / "runtime",
        system="Linux",
        machine="x86_64",
    )

    assert result.state is QwenReadinessState.UNSUPPORTED
    assert result.no_fallback_attempted is True
    assert "qwen_asr" not in set(sys.modules) - before


def test_static_readiness_distinguishes_missing_and_invalid_runtime(tmp_path: Path) -> None:
    runtime, spec = _runtime(tmp_path)
    missing = inspect_qwen_readiness(
        spec,
        store_root=tmp_path / "store",
        runtime_root=runtime,
        system="Darwin",
        machine="arm64",
    )
    assert missing.state is QwenReadinessState.MISSING

    (runtime / ".runtime-receipt.json").chmod(0o644)
    invalid = inspect_qwen_readiness(
        spec,
        store_root=tmp_path / "store",
        runtime_root=runtime,
        system="Darwin",
        machine="arm64",
    )
    assert invalid.state is QwenReadinessState.INVALID
    assert "private" in invalid.detail


@pytest.mark.parametrize(
    ("artifact_state", "expected"),
    [
        (ArtifactState.PROVISIONED, QwenReadinessState.PROVISIONED),
        (ArtifactState.INVALID, QwenReadinessState.INVALID),
    ],
)
def test_static_readiness_maps_artifact_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    artifact_state: ArtifactState,
    expected: QwenReadinessState,
) -> None:
    runtime, spec = _runtime(tmp_path)

    class FakeStore:
        def __init__(self, root: Path) -> None:
            self.root = root

        def inspect(self, selected):
            return ArtifactInspection(
                selected.key,
                artifact_state,
                selected.model_path(self.root),
                "fixture detail",
            )

    monkeypatch.setattr("soca.asr.qwen_readiness.QwenArtifactStore", FakeStore)
    result = inspect_qwen_readiness(
        spec,
        store_root=tmp_path / "store",
        runtime_root=runtime,
        system="Darwin",
        machine="arm64",
    )

    assert result.state is expected
    if expected is QwenReadinessState.PROVISIONED:
        assert "service stopped" in result.detail
