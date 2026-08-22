from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType


def _builder_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_desktop_sidecar.py"
    spec = importlib.util.spec_from_file_location("build_desktop_sidecar", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_sidecar_uses_host_triple_and_explicit_dependency_closure(
    monkeypatch, tmp_path: Path
) -> None:
    builder = _builder_module()
    entry = tmp_path / "desktop" / "sidecar" / "soca_engine.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("print('entry')\n", encoding="utf-8")
    monkeypatch.setattr(builder, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(builder, "ENTRY_POINT", entry)
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[:3] == ["rustc", "--print", "host-tuple"]:
            return subprocess.CompletedProcess(command, 0, "x86_64-unknown-linux-gnu\n", "")
        dist = Path(command[command.index("--distpath") + 1])
        dist.mkdir(parents=True)
        suffix = ".exe" if builder.os.name == "nt" else ""
        (dist / f"{builder.SIDECAR_BASENAME}{suffix}").write_bytes(b"frozen-engine")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(builder.subprocess, "run", fake_run)
    destination = builder.build_sidecar(tmp_path / "binaries")

    assert destination.name == builder.sidecar_filename("x86_64-unknown-linux-gnu")
    assert destination.read_bytes() == b"frozen-engine"
    pyinstaller = commands[1]
    assert pyinstaller[:3] == [sys.executable, "-m", "PyInstaller"]
    assert ["--collect-all", "soca"] == pyinstaller[
        pyinstaller.index("--collect-all") : pyinstaller.index("--collect-all") + 2
    ]
    assert ["--collect-binaries", "llama_cpp"] == pyinstaller[
        pyinstaller.index("--collect-binaries") : pyinstaller.index("--collect-binaries") + 2
    ]
    assert ["--copy-metadata", "torchcodec"] == pyinstaller[
        pyinstaller.index("--copy-metadata") : pyinstaller.index("--copy-metadata") + 2
    ]
    assert str(entry) == pyinstaller[-1]


def test_rust_target_triple_falls_back_to_verbose_output(monkeypatch) -> None:
    builder = _builder_module()
    calls = 0

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(command, 1, "", "unsupported")
        return subprocess.CompletedProcess(command, 0, "host: aarch64-apple-darwin\n", "")

    monkeypatch.setattr(builder.subprocess, "run", fake_run)

    assert builder.rust_target_triple() == "aarch64-apple-darwin"
