from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from scripts import provision_qwen_asr as provision
from soca.asr.qwen_artifacts import QWEN_RELEASE_ARTIFACT
from soca.asr.result import ASRResult


def test_worker_runtime_verification_requires_private_exact_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime = tmp_path / "runtime"
    python = runtime / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    lock = runtime / "uv.lock"
    lock.write_text("locked", encoding="utf-8")
    receipt = runtime / ".runtime-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "python": provision.EXPECTED_PYTHON,
                "uv": provision.EXPECTED_UV,
                "lock_sha256": provision._sha256(lock),
                "soca_wheel_sha256": "a" * 64,
                "environment": str(runtime / ".venv"),
            }
        ),
        encoding="utf-8",
    )
    receipt.chmod(0o600)
    monkeypatch.setattr(provision, "RUNTIME_PROJECT", runtime)
    monkeypatch.setattr(provision, "RUNTIME_LOCK", lock)
    monkeypatch.setattr(provision, "RUNTIME_RECEIPT", receipt)
    monkeypatch.setattr(provision, "RUNTIME_PYTHON", python)
    monkeypatch.setattr(provision.shutil, "which", lambda _: "/usr/local/bin/uvx")
    completed = MagicMock(returncode=0, stdout=f"{provision.EXPECTED_PYTHON}\n")
    monkeypatch.setattr(provision.subprocess, "run", lambda *args, **kwargs: completed)

    result = provision.verify_worker_runtime()

    assert result["lock_sha256"] == provision._sha256(lock)
    receipt.chmod(0o644)
    with pytest.raises(provision.QwenProvisionCommandError, match="not private"):
        provision.verify_worker_runtime()


def test_health_probe_uses_local_path_offline_and_closes_worker(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "voice.wav"
    sf.write(audio_path, np.ones(1_600, dtype=np.float32) * 0.01, 16_000)
    clients: list[object] = []

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.closed = False
            clients.append(self)

        def transcribe(self, audio, max_new_tokens):
            assert audio.shape == (1_600,)
            assert max_new_tokens == 128
            return ASRResult(
                text="bản ghi thật",
                latency_ms=50.0,
                audio_duration_ms=100.0,
                rtf=0.5,
                avg_logprob=-0.1,
                avg_logprob_reliable=True,
            )

        def close(self) -> None:
            self.closed = True

    model_path = tmp_path / "model"
    model_path.mkdir()
    result = provision.build_health_probe(
        audio_path,
        QWEN_RELEASE_ARTIFACT,
        client_factory=FakeClient,  # type: ignore[arg-type]
    )(model_path)

    client = clients[0]
    assert client.kwargs["launch"].model_path == model_path  # type: ignore[attr-defined]
    assert client.kwargs["process_environment"] == {  # type: ignore[attr-defined]
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    assert client.closed is True  # type: ignore[attr-defined]
    assert result["transcript"] == "bản ghi thật"
    assert result["audio_sha256"] == provision._sha256(audio_path)


def test_install_defaults_to_pinned_mirror_and_never_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(provision, "verify_worker_runtime", lambda: {})
    monkeypatch.setenv("HF_TOKEN", "private-token-must-not-leak")

    exit_code = provision.main(
        [
            "--store-root",
            str(tmp_path / "store"),
            "install",
            "--artifact",
            "release",
            "--health-audio",
            str(tmp_path / "voice.wav"),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 2
    assert "MirrorNotPinned" in output
    assert "private-token-must-not-leak" not in output


def test_refresh_cli_reissues_receipt_with_health_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    receipt = type("Receipt", (), {"model_path": str(tmp_path / "model")})()
    captured: dict[str, object] = {}

    class FakeStore:
        def refresh_receipt(self, spec, *, source_kind, health_probe, runtime_lock):
            captured.update(
                {
                    "key": spec.key,
                    "source_kind": source_kind.value,
                    "health": health_probe(tmp_path / "model"),
                    "runtime_lock": runtime_lock,
                }
            )
            return receipt

    monkeypatch.setattr(provision, "verify_worker_runtime", lambda: {"python": "3.11.14"})
    monkeypatch.setattr(provision, "QwenArtifactStore", lambda _root: FakeStore())
    monkeypatch.setattr(
        provision,
        "build_health_probe",
        lambda _audio, _spec: lambda _path: {"transcript": "ok"},
    )

    exit_code = provision.main(
        [
            "--store-root",
            str(tmp_path / "store"),
            "refresh",
            "--artifact",
            "release",
            "--source",
            "upstream",
            "--health-audio",
            str(tmp_path / "voice.wav"),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["receipt_refreshed"] is True
    assert captured == {
        "key": QWEN_RELEASE_ARTIFACT.key,
        "source_kind": "upstream",
        "health": {"transcript": "ok"},
        "runtime_lock": provision.RUNTIME_LOCK,
    }


def test_inspect_is_static_and_reports_both_missing_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = provision.main(["--store-root", str(tmp_path / "store"), "inspect", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert [item["state"] for item in payload["artifacts"]] == ["missing", "missing"]


def test_deep_verify_requires_explicit_real_health_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(provision, "verify_worker_runtime", lambda: {})

    exit_code = provision.main(
        [
            "--store-root",
            str(tmp_path / "store"),
            "verify",
            "--artifact",
            "release",
            "--deep",
        ]
    )

    assert exit_code == 2
    assert "requires --health-audio" in capsys.readouterr().out
