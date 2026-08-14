from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from local import config as cfg

_GIT_TIMEOUT_S = 5


def _git(*args: str) -> str | None:
    """Run a git command inside the repo, or return None if git cannot answer."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cfg.REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def run_provenance(**extra: Any) -> dict[str, Any]:
    """Describe the tree a result was produced from.

    A conversational result is a function of the endpoint constants as much as of
    the audio, and those constants live in source. Without the revision stamped
    into the artifact, a later tuning commit silently invalidates the published
    number and nothing in the file says so. ``dirty`` matters just as much: a run
    from an uncommitted tree cannot be reproduced by anybody, including us.
    """
    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    provenance: dict[str, Any] = {
        "commit": head,
        "dirty": None if status is None else bool(status),
        "created_at": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
    }
    provenance.update(extra)
    return provenance


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_set_identity(paths: tuple[Path, ...]) -> dict[str, Any]:
    """Digest an ordered file set without exposing private file contents."""
    if not paths:
        raise ValueError("file identity requires at least one file")
    digest = hashlib.sha256()
    total_bytes = 0
    for index, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        item_hash = file_sha256(path)
        digest.update(f"{index}:{size}:{item_hash}\n".encode())
        total_bytes += size
    return {
        "file_count": len(paths),
        "total_bytes": total_bytes,
        "content_sha256": digest.hexdigest(),
    }


def package_versions(names: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"required package is not installed: {name}") from exc
    return versions


def config_snapshot(config: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    """Record the settings that govern a result so drift is visible in the artifact."""
    if not is_dataclass(config) or isinstance(config, type):
        raise TypeError(f"expected a dataclass instance, got {type(config)!r}")
    values = asdict(config)
    missing = [field for field in fields if field not in values]
    if missing:
        raise KeyError(f"{type(config).__name__} has no field(s): {', '.join(missing)}")
    return {field: values[field] for field in fields}
