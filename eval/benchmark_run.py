from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _git_value(*arguments: str) -> str:
    try:
        return subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(path)


def run_logged_command(
    command: tuple[str, ...],
    *,
    run_dir: Path,
    family: str,
    cwd: Path | None = None,
) -> int:
    if not command:
        raise ValueError("benchmark command must not be empty")
    if not family.strip():
        raise ValueError("benchmark family must not be empty")
    resolved_cwd = (cwd or Path.cwd()).resolve()
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    run_dir.chmod(0o700)
    log_path = run_dir / "run.log"
    provenance_path = run_dir / "run.json"
    started_at = datetime.now(UTC)
    environment = os.environ.copy()
    environment["SOCA_BENCHMARK_RUN_DIR"] = str(run_dir.resolve())
    process = subprocess.Popen(
        command,
        cwd=resolved_cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    with log_path.open("w", encoding="utf-8") as log:
        log_path.chmod(0o600)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
    exit_code = process.wait()
    finished_at = datetime.now(UTC)
    status = _git_value("status", "--porcelain=v1", "--untracked-files=all")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "family": family,
        "command": list(command),
        "cwd": str(resolved_cwd),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "exit_code": exit_code,
        "source": {
            "commit": _git_value("rev-parse", "HEAD"),
            "dirty": status not in {"", "unknown"},
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "inputs": {
            "uv_lock_sha256": _file_sha256(resolved_cwd / "uv.lock"),
            "retrieval_source_lock_sha256": _file_sha256(
                resolved_cwd / "data/benchmarks/retrieval/sources.lock.json"
            ),
            "retrieval_provision_manifest_sha256": _file_sha256(
                resolved_cwd / "data/benchmarks/retrieval/provisioned-manifest.json"
            ),
        },
    }
    _write_private_json(provenance_path, payload)
    return exit_code
