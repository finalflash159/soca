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
    calibration = tmp_path / "data" / "asr"
    calibration.mkdir(parents=True)
    monkeypatch.setattr(builder, "ASR_CALIBRATION_DATA", calibration)
    smart_turn_model = tmp_path / "models" / "smart-turn-v3-onnx" / "smart-turn-v3.2-cpu.onnx"
    smart_turn_model.parent.mkdir(parents=True)
    smart_turn_model.write_bytes(b"smart-turn")
    monkeypatch.setattr(builder, "SMART_TURN_MODEL", smart_turn_model)
    monkeypatch.setattr(builder, "copy_linux_cuda_runtime_libraries", lambda runtime: None)
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
    collected_modules = [
        pyinstaller[index : index + 2]
        for index, value in enumerate(pyinstaller)
        if value == "--collect-all"
    ]
    assert ["--collect-all", "soca"] in collected_modules
    assert ["--collect-all", "silero_vad"] in collected_modules
    excluded_modules = [
        pyinstaller[index : index + 2]
        for index, value in enumerate(pyinstaller)
        if value == "--exclude-module"
    ]
    assert ["--exclude-module", "librosa"] in excluded_modules
    assert ["--exclude-module", "numba"] in excluded_modules
    assert ["--exclude-module", "llvmlite"] in excluded_modules
    assert ["--add-data", f"{calibration}{builder.os.pathsep}data/asr"] == pyinstaller[
        pyinstaller.index("--add-data") : pyinstaller.index("--add-data") + 2
    ]
    add_data = [
        pyinstaller[index : index + 2]
        for index, value in enumerate(pyinstaller)
        if value == "--add-data"
    ]
    assert [
        "--add-data",
        f"{smart_turn_model}{builder.os.pathsep}data/smart-turn-v3-onnx",
    ] in add_data
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
    excluded_modules = [
        pyinstaller[index : index + 2]
        for index, value in enumerate(pyinstaller)
        if value == "--exclude-module"
    ]
    assert ["--exclude-module", "torchcodec"] in excluded_modules
    assert str(entry) == pyinstaller[-1]


def test_linux_cuda_runtime_is_copied_from_its_distribution(monkeypatch, tmp_path: Path) -> None:
    builder = _builder_module()
    monkeypatch.setattr(builder.sys, "platform", "linux")
    library = tmp_path / "site-packages" / "nvidia" / "cu13" / "lib" / "libcudart.so.13"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"cuda")

    class Distribution:
        files = (Path("nvidia/cu13/lib/libcudart.so.13"),)

        def locate_file(self, file: Path) -> Path:
            assert file == self.files[0]
            return library

    monkeypatch.setattr(builder.metadata, "distribution", lambda name: Distribution())

    assert builder.cuda_runtime_libraries() == [library]

    runtime = tmp_path / "runtime"
    builder.copy_linux_cuda_runtime_libraries(runtime)
    assert (runtime / "_internal" / "libcudart.so.13").read_bytes() == b"cuda"


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
    cuda_lib = internal
    (cuda_lib / "libcudart.so.13").write_bytes(b"cuda")

    builder.link_linux_torchaudio_dependencies(tmp_path)

    for library in builder.TORCHAUDIO_TORCH_LIBRARIES:
        linked = torchaudio_lib / library
        assert linked.is_symlink()
        assert linked.resolve() == torch_lib / library
    cuda_link = torchaudio_lib / "libcudart.so.13"
    assert cuda_link.is_file()
    assert not cuda_link.is_symlink()
    assert cuda_link.read_bytes() == b"cuda"
    assert not (torchaudio_lib / "libtorch_cuda.so").exists()
