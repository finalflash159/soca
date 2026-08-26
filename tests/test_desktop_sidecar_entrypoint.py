from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from soca.config import secret_store
from soca.core.workflow.protocol import CURRENT_PROTOCOL_VERSION


def _entrypoint() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "desktop" / "sidecar" / "soca_engine.py"
    spec = importlib.util.spec_from_file_location("soca_desktop_sidecar", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_staging_hello_is_immediate_and_protocol_compatible(capsys) -> None:
    entrypoint = _entrypoint()

    entrypoint.emit_staging_hello(profile=None, no_model=True)

    frame = json.loads(capsys.readouterr().out)
    assert frame == {
        "event": "hello",
        "version": CURRENT_PROTOCOL_VERSION,
        "protocol_version": CURRENT_PROTOCOL_VERSION,
        "supported_versions": [CURRENT_PROTOCOL_VERSION],
        "profile": "initializing",
        "no_model": True,
        "stack": {"llm": "initializing"},
    }
    assert entrypoint.PROTOCOL_VERSION == CURRENT_PROTOCOL_VERSION


def test_entrypoint_accepts_the_engine_contract() -> None:
    entrypoint = _entrypoint()

    parsed = entrypoint.parse_args(
        ["engine", "--no-model", "--session-persistence", "local_resumable"]
    )

    assert parsed.command == "engine"
    assert parsed.no_model is True
    assert parsed.session_persistence == "local_resumable"


def test_run_dispatches_frozen_multiprocessing_before_the_engine(monkeypatch) -> None:
    entrypoint = _entrypoint()
    calls: list[str] = []

    monkeypatch.setattr(entrypoint.multiprocessing, "freeze_support", lambda: calls.append("freeze"))
    monkeypatch.setattr(entrypoint, "main", lambda: calls.append("main") or 0)

    assert entrypoint.run() == 0
    assert calls == ["freeze", "main"]


def test_keyring_helper_uses_only_its_internal_contract(monkeypatch, capsys) -> None:
    entrypoint = _entrypoint()

    class FakeKeyring:
        def get_password(self, service: str, username: str) -> str | None:
            assert service == "soca-llm"
            assert username == "openai"
            return "key-from-keyring"

        def set_password(self, service: str, username: str, value: str) -> None:
            raise AssertionError("get operation must not write")

    monkeypatch.setattr(secret_store, "_load_keyring", lambda: FakeKeyring())
    monkeypatch.setattr(
        entrypoint.sys,
        "stdin",
        io.StringIO('{"service":"soca-llm","username":"openai"}'),
    )

    assert entrypoint.run_keyring_helper("get") == 0
    assert json.loads(capsys.readouterr().out) == {"value": "key-from-keyring"}


def test_keyring_backend_turns_helper_timeout_into_typed_failure(monkeypatch) -> None:
    entrypoint = _entrypoint()
    calls: list[tuple[str, ...]] = []

    def timeout(*args, **kwargs):
        calls.append(tuple(args[0]))
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(entrypoint.subprocess, "run", timeout)
    backend = entrypoint.SidecarKeyringBackend(Path("/tmp/soca-engine"), timeout_seconds=0.75)

    try:
        backend.get_password("soca-llm", "openai")
    except RuntimeError as exc:
        assert str(exc) == "Desktop keyring helper timed out"
    else:
        raise AssertionError("expected the timed-out helper failure")
    assert calls == [("/tmp/soca-engine", "keyring", "--keyring-operation", "get")]
    with pytest.raises(RuntimeError, match="helper is unavailable"):
        backend.get_password("soca-llm", "groq")
    assert len(calls) == 1
