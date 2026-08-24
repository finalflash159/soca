from __future__ import annotations

import importlib.util
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


def test_build_sidecar_uses_one_directory_runtime_and_explicit_dependency_closure(
    monkeypatch, tmp_path: Path
) -> None:
    builder = _builder_module()
    entry = tmp_path / "desktop" / "sidecar" / "soca_engine.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("print('entry')\n", encoding="utf-8")
    monkeypatch.setattr(builder, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(builder, "ENTRY_POINT", entry)
    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> None:
        commands.append(command)
        dist = Path(command[command.index("--distpath") + 1])
        runtime = dist / builder.SIDECAR_BASENAME
        runtime.mkdir(parents=True)
        suffix = ".exe" if builder.os.name == "nt" else ""
        (runtime / f"{builder.SIDECAR_BASENAME}{suffix}").write_bytes(b"frozen-engine")

    monkeypatch.setattr(builder.subprocess, "run", fake_run)
    destination = builder.build_sidecar(tmp_path / "resources")

    assert destination.name == f"{builder.SIDECAR_BASENAME}{'.exe' if builder.os.name == 'nt' else ''}"
    assert destination.parent.name == builder.SIDECAR_BASENAME
    assert destination.read_bytes() == b"frozen-engine"
    pyinstaller = commands[0]
    assert pyinstaller[:3] == [sys.executable, "-m", "PyInstaller"]
    assert "--onedir" in pyinstaller
    assert "--onefile" not in pyinstaller
    assert ["--collect-all", "soca"] == pyinstaller[
        pyinstaller.index("--collect-all") : pyinstaller.index("--collect-all") + 2
    ]
    assert ["--collect-binaries", "llama_cpp"] == pyinstaller[
        pyinstaller.index("--collect-binaries") : pyinstaller.index("--collect-binaries") + 2
    ]
    binary_collections = [
        pyinstaller[index : index + 2]
        for index, value in enumerate(pyinstaller)
        if value == "--collect-binaries"
    ]
    assert ["--collect-binaries", "torch"] in binary_collections
    assert ["--collect-binaries", "torchaudio"] in binary_collections
    assert ["--collect-binaries", "nvidia.cuda_runtime"] in binary_collections
    assert ["--copy-metadata", "torchcodec"] == pyinstaller[
        pyinstaller.index("--copy-metadata") : pyinstaller.index("--copy-metadata") + 2
    ]
    assert str(entry) == pyinstaller[-1]


def test_linux_torchaudio_links_resolve_torch_shared_libraries(
    monkeypatch, tmp_path: Path
) -> None:
    builder = _builder_module()
    monkeypatch.setattr(builder.sys, "platform", "linux")
    internal = tmp_path / "_internal"
    torch_lib = internal / "torch" / "lib"
    torchaudio_lib = internal / "torchaudio" / "lib"
    torch_lib.mkdir(parents=True)
    torchaudio_lib.mkdir(parents=True)
    (torch_lib / "libc10.so").write_bytes(b"torch")
    (torch_lib / "libtorch.so").write_bytes(b"torch")
    (torch_lib / "libtorch_cpu.so").write_bytes(b"torch")
    (torch_lib / "libtorch_global_deps.so").write_bytes(b"torch")
    (torch_lib / "libtorch_cuda.so").write_bytes(b"unused")
    cuda_lib = internal / "nvidia" / "cuda_runtime" / "lib"
    cuda_lib.mkdir(parents=True)
    (cuda_lib / "libcudart.so.13").write_bytes(b"cuda")

    builder.link_linux_torchaudio_dependencies(tmp_path)

    for library in builder.TORCHAUDIO_TORCH_LIBRARIES:
        linked = torchaudio_lib / library
        assert linked.is_symlink()
        assert linked.resolve() == torch_lib / library
    cuda_link = torchaudio_lib / "libcudart.so.13"
    assert cuda_link.is_symlink()
    assert cuda_link.resolve() == cuda_lib / "libcudart.so.13"
    assert not (torchaudio_lib / "libtorch_cuda.so").exists()
