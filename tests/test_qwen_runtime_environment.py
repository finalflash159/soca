from __future__ import annotations

import hashlib
import json
import stat
import tomllib
from pathlib import Path

import scripts.provision_qwen_runtime as provisioner
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


def test_worker_sbom_matches_the_minimal_runtime() -> None:
    sbom = json.loads((RUNTIME_ROOT / "sbom.cdx.json").read_text(encoding="utf-8"))
    components = {component["name"] for component in sbom["components"]}

    assert "qwen-asr" in components
    assert not components.intersection({"flask", "gradio", "qwen-omni-utils", "sox"})


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
