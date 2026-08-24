from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import tomllib
from pathlib import Path

import pytest

import scripts.provision_qwen_runtime as provisioner
from soca.asr.qwen_runtime import (
    QWEN_RUNTIME_ROOT_ENV,
    default_qwen_runtime_root,
    default_qwen_venv_python,
)
from soca.asr.qwen_service_client import QWEN_VENV_PYTHON

RUNTIME_ROOT = Path("runtime/qwen-asr")


def test_worker_runtime_is_exact_and_excludes_demo_dependencies() -> None:
    project = tomllib.loads((RUNTIME_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    uv = project["tool"]["uv"]

    assert project["project"]["requires-python"] == "==3.11.*"
    assert uv["required-version"] == "==0.11.16"
    assert set(uv["exclude-dependencies"]) == {
        "flask",
        "gradio",
        "pytz",
        "qwen-omni-utils",
        "sox",
    }
    assert (RUNTIME_ROOT / "uv.lock").is_file()
    assert QWEN_VENV_PYTHON == RUNTIME_ROOT.resolve() / ".venv/bin/python"


def test_worker_runtime_honors_an_explicit_absolute_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "qwen-runtime"
    monkeypatch.setenv(QWEN_RUNTIME_ROOT_ENV, str(runtime))

    assert default_qwen_runtime_root() == runtime
    assert default_qwen_venv_python() == runtime / ".venv" / "bin" / "python"


def test_worker_runtime_rejects_relative_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(QWEN_RUNTIME_ROOT_ENV, "runtime/qwen-asr")

    with pytest.raises(ValueError, match="absolute"):
        default_qwen_runtime_root()


def test_worker_sbom_matches_the_minimal_runtime() -> None:
    sbom = json.loads((RUNTIME_ROOT / "sbom.cdx.json").read_text(encoding="utf-8"))
    components = {component["name"] for component in sbom["components"]}

    assert "qwen-asr" in components
    assert not components.intersection({"flask", "gradio", "qwen-omni-utils", "sox"})


def test_runtime_benchmark_evidence_is_reproducible_release_evidence() -> None:
    evidence = json.loads(
        Path("research/qwen_runtime_layouts/results/macos-arm64.json").read_text(
            encoding="utf-8"
        )
    )

    assert evidence["run_type"] == "release_benchmark"
    assert evidence["artifact"]["revision"]
    assert evidence["data"]["manifest_sha256"]
    assert all(item["lock_sha256"] for item in evidence["results"])
    assert evidence["raw_logs"]["committed"] is False
    assert "failures" in evidence
    assert evidence["decision"]["status"] == "accepted"


def test_artifacts_bind_to_the_committed_runtime_lock() -> None:
    lock_digest = hashlib.sha256((RUNTIME_ROOT / "uv.lock").read_bytes()).hexdigest()
    for name in ("qwen3_asr_0_6b.json", "qwen3_asr_1_7b.json"):
        manifest = json.loads(Path("soca/asr/artifacts", name).read_text(encoding="utf-8"))
        assert manifest["runtime_lock_digest"] == lock_digest


def test_runtime_receipt_is_owner_private(tmp_path: Path, monkeypatch) -> None:
    receipt_path = tmp_path / "receipt.json"
    monkeypatch.setattr(provisioner, "RECEIPT_PATH", receipt_path)

    provisioner._write_private_receipt({"schema_version": 1})

    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == {"schema_version": 1}


def test_external_commands_have_an_explicit_timeout(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(provisioner.subprocess, "run", fake_run)

    provisioner._run(["uv", "sync"], timeout_s=123.0)

    assert observed["timeout"] == 123.0


def test_external_command_timeout_is_typed(monkeypatch) -> None:
    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(provisioner.subprocess, "run", fake_run)

    with pytest.raises(
        provisioner.QwenRuntimeProvisionError,
        match="timed out after 45",
    ):
        provisioner._run(["uv", "sync"], timeout_s=45.0)
