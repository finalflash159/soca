"""Build the native, self-contained Python engine consumed by Tauri.

Tauri requires an ``externalBin`` filename suffixed with the Rust target triple.
PyInstaller is invoked with the active interpreter, so each operating system
produces its own sidecar instead of pretending to cross-build native extensions.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

SIDECAR_BASENAME = "soca-engine"
REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRY_POINT = REPO_ROOT / "desktop" / "sidecar" / "soca_engine.py"


def rust_target_triple() -> str:
    """Return the host triple accepted by Tauri's ``externalBin`` convention."""

    primary = subprocess.run(
        ["rustc", "--print", "host-tuple"],
        check=False,
        capture_output=True,
        text=True,
    )
    target = primary.stdout.strip()
    if primary.returncode == 0 and target:
        return target

    # ``--print host-tuple`` was added after older still-supported Rust toolchains.
    fallback = subprocess.run(
        ["rustc", "-Vv"], check=True, capture_output=True, text=True
    )
    for line in fallback.stdout.splitlines():
        if line.startswith("host: "):
            return line.removeprefix("host: ").strip()
    raise RuntimeError("rustc did not report a host target triple")


def sidecar_filename(target_triple: str) -> str:
    suffix = ".exe" if os.name == "nt" else ""
    return f"{SIDECAR_BASENAME}-{target_triple}{suffix}"


def pyinstaller_command(*, dist: Path, work: Path, spec: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
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

    target = rust_target_triple()
    build_root = REPO_ROOT / "build" / "desktop-sidecar"
    dist = build_root / "dist"
    work = build_root / "work"
    spec = build_root / "spec"
    command = pyinstaller_command(dist=dist, work=work, spec=spec)
    environment = {**os.environ, "PYTHONNOUSERSITE": "1"}
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=environment)

    suffix = ".exe" if os.name == "nt" else ""
    produced = dist / f"{SIDECAR_BASENAME}{suffix}"
    if not produced.is_file():
        raise RuntimeError(f"PyInstaller completed without producing {produced}")

    output.mkdir(parents=True, exist_ok=True)
    destination = output / sidecar_filename(target)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(produced, temporary)
    temporary.chmod(temporary.stat().st_mode | stat.S_IXUSR)
    temporary.replace(destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="directory containing the Tauri externalBin files",
    )
    args = parser.parse_args()
    destination = build_sidecar(args.output.expanduser().resolve())
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
