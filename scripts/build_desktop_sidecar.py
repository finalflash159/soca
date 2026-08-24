"""Build the native, self-contained Python engine consumed by Tauri.

The engine is a PyInstaller one-directory runtime bundled as a Tauri resource.
Keeping its dependency tree beside the executable avoids extracting hundreds of
megabytes into a temporary directory on every cold launch. PyInstaller still
runs through the active native interpreter: each operating system builds its
own runtime and native extensions are never cross-compiled.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

SIDECAR_BASENAME = "soca-engine"
REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRY_POINT = REPO_ROOT / "desktop" / "sidecar" / "soca_engine.py"


def pyinstaller_command(*, dist: Path, work: Path, spec: Path) -> list[str]:
    return [
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
        # llama-cpp-python locates libllama relative to its package at runtime;
        # its native libraries are not visible through Python imports alone.
        "--collect-binaries",
        "llama_cpp",
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


def build_sidecar(output: Path) -> Path:
    if not ENTRY_POINT.is_file():
        raise RuntimeError(f"missing frozen sidecar entry point: {ENTRY_POINT}")

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

    output.mkdir(parents=True, exist_ok=True)
    destination = output / SIDECAR_BASENAME
    temporary = output / f".{SIDECAR_BASENAME}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(produced, temporary, copy_function=shutil.copy2)
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
