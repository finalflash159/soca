"""Build the PyInstaller runtime embedded in the desktop application."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

SIDECAR_BASENAME = "soca-engine"
REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRY_POINT = REPO_ROOT / "desktop" / "sidecar" / "soca_engine.py"
ASR_CALIBRATION_DATA = REPO_ROOT / "data" / "asr"
TORCHAUDIO_TORCH_LIBRARIES = (
    "libc10.so",
    "libtorch.so",
    "libtorch_cpu.so",
    "libtorch_global_deps.so",
)


def cuda_runtime_libraries() -> list[Path]:
    """Locate the CUDA runtime files installed alongside Linux Torch."""
    if not sys.platform.startswith("linux"):
        return []

    try:
        distribution = metadata.distribution("nvidia-cuda-runtime")
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "Linux desktop sidecar requires the nvidia-cuda-runtime distribution"
        ) from error

    libraries = [
        Path(distribution.locate_file(file))
        for file in distribution.files or ()
        if len(file.parts) >= 4
        and file.parts[0] == "nvidia"
        and file.parts[-2] == "lib"
        and file.name.startswith("libcudart.so")
    ]
    if not libraries:
        raise RuntimeError("nvidia-cuda-runtime does not contain a libcudart shared library")

    return libraries


def copy_linux_cuda_runtime_libraries(runtime: Path) -> None:
    """Place CUDA beside the frozen interpreter before Tauri copies the sidecar."""
    if not sys.platform.startswith("linux"):
        return

    internal = runtime / "_internal"
    internal.mkdir(parents=True, exist_ok=True)
    for library in cuda_runtime_libraries():
        if not library.is_file():
            raise RuntimeError(f"missing CUDA runtime library: {library}")
        shutil.copy2(library, internal / library.name)


def link_linux_torchaudio_dependencies(runtime: Path) -> None:
    """Expose Torch's shared libraries to Torchaudio and Linux AppImage tooling."""
    if not sys.platform.startswith("linux"):
        return

    internal = runtime / "_internal"
    torch_lib = internal / "torch" / "lib"
    torchaudio_lib = internal / "torchaudio" / "lib"
    if not torch_lib.is_dir() or not torchaudio_lib.is_dir():
        return

    library_directories = [torch_lib, internal, *internal.glob("nvidia/*/lib")]
    for library_directory in library_directories:
        library_names = (
            TORCHAUDIO_TORCH_LIBRARIES
            if library_directory == torch_lib
            else (
                tuple(library.name for library in library_directory.glob("libcudart.so*"))
                if library_directory == internal
                else tuple(library.name for library in library_directory.glob("*.so*"))
            )
        )
        for library_name in library_names:
            library = library_directory / library_name
            if not library.is_file():
                continue
            destination = torchaudio_lib / library_name
            if destination.exists() or destination.is_symlink():
                continue
            destination.symlink_to(os.path.relpath(library, start=torchaudio_lib))


def pyinstaller_command(*, dist: Path, work: Path, spec: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        SIDECAR_BASENAME,
        "--paths",
        str(REPO_ROOT),
        # ``soca`` loads providers and runtime profiles dynamically. Collecting its
        # submodules is the explicit closure, rather than relying on a developer's
        # editable checkout at runtime.
        "--collect-all",
        "soca",
        # Runtime readiness and the production ASR guard both load the
        # committed calibration JSON beside the frozen package root. Without
        # this data closure a desktop build can find every model file yet
        # truthfully reject Voice as uncalibrated.
        "--add-data",
        f"{ASR_CALIBRATION_DATA}{os.pathsep}data/asr",
        # llama-cpp-python locates libllama relative to its package at runtime;
        # its native libraries are not visible through Python imports alone.
        "--collect-binaries",
        "llama_cpp",
        # Torchaudio loads Torch shared objects dynamically. Make the native
        # closure explicit so the frozen runtime and AppImage bundle contain
        # libc10/libtorch instead of depending on a build runner's environment.
        "--collect-binaries",
        "torch",
        "--collect-binaries",
        "torchaudio",
        # transformers checks this distribution's version during ASR imports.
        # Do not collect torchcodec's bundled libpython: it is a separate
        # runtime and would conflict with the interpreter frozen by PyInstaller.
        "--copy-metadata",
        "torchcodec",
        "--distpath",
        str(dist),
        "--workpath",
        str(work),
        "--specpath",
        str(spec),
        str(ENTRY_POINT),
    ]
    return command


def build_sidecar(output: Path) -> Path:
    if not ENTRY_POINT.is_file():
        raise RuntimeError(f"missing frozen sidecar entry point: {ENTRY_POINT}")
    if not ASR_CALIBRATION_DATA.is_dir():
        raise RuntimeError(f"missing ASR calibration data: {ASR_CALIBRATION_DATA}")

    build_root = REPO_ROOT / "build" / "desktop-sidecar"
    dist = build_root / "dist"
    work = build_root / "work"
    spec = build_root / "spec"
    command = pyinstaller_command(dist=dist, work=work, spec=spec)
    environment = {**os.environ, "PYTHONNOUSERSITE": "1"}
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=environment)

    produced = dist / SIDECAR_BASENAME
    executable = produced / f"{SIDECAR_BASENAME}{'.exe' if os.name == 'nt' else ''}"
    if not executable.is_file():
        raise RuntimeError(f"PyInstaller completed without producing {produced}")
    copy_linux_cuda_runtime_libraries(produced)
    link_linux_torchaudio_dependencies(produced)

    output.mkdir(parents=True, exist_ok=True)
    destination = output / SIDECAR_BASENAME
    temporary = output / f".{SIDECAR_BASENAME}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(produced, temporary, copy_function=shutil.copy2, symlinks=True)
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)
    return destination / executable.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="directory containing the Tauri resource runtime directory",
    )
    args = parser.parse_args()
    destination = build_sidecar(args.output.expanduser().resolve())
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
