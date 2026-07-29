from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalRunPaths:
    run_dir: Path
    json_path: Path
    md_path: Path
    latest_json_path: Path
    latest_md_path: Path


@dataclass(frozen=True)
class EvalArtifactMetadata:
    """Reproducibility envelope shared by evaluation artifacts.

    Quality decisions must carry the exact source revision and dataset digest.
    The envelope deliberately contains no prompt content beyond file paths and
    hashes, so it is safe to publish alongside a report.
    """

    schema_version: str
    suite: str
    git_commit: str
    generated_at_utc: str
    python_version: str
    platform: str
    source_dirty: bool
    source_state_digest: str
    data_files: tuple[dict[str, Any], ...]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite": self.suite,
            "git_commit": self.git_commit,
            "generated_at_utc": self.generated_at_utc,
            "environment": {
                "python": self.python_version,
                "platform": self.platform,
            },
            "source": {
                "commit": self.git_commit,
                "dirty": self.source_dirty,
                "state_digest": self.source_state_digest,
            },
            "data_files": list(self.data_files),
            "config": self.config,
        }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_source_state() -> tuple[bool, str]:
    """Return dirty state and a digest of tracked/untracked source changes."""

    try:
        diff = subprocess.check_output(
            ["git", "diff", "HEAD", "--binary"],
            stderr=subprocess.DEVNULL,
        )
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            stderr=subprocess.DEVNULL,
        )
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            stderr=subprocess.DEVNULL,
        )
        untracked_digests: list[bytes] = []
        for raw_path in untracked.split(b"\0"):
            if not raw_path:
                continue
            file_path = Path(raw_path.decode("utf-8", errors="surrogateescape"))
            if file_path.is_file():
                untracked_digests.append(raw_path + b"\0" + bytes.fromhex(_sha256(file_path)))
        state = b"\0".join((status, diff, *sorted(untracked_digests)))
        digest = hashlib.sha256(state).hexdigest()
        return bool(status or diff or untracked_digests), digest
    except (OSError, subprocess.CalledProcessError):
        return True, "unknown"


def make_eval_artifact_metadata(
    *,
    suite: str,
    data_files: tuple[Path, ...],
    config: dict[str, Any] | None = None,
    schema_version: str = "soca-eval-artifact-v1",
) -> EvalArtifactMetadata:
    """Capture provenance required for a reproducible quality result."""

    source_dirty, source_state_digest = _git_source_state()
    files = tuple(
        {
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in data_files
    )
    return EvalArtifactMetadata(
        schema_version=schema_version,
        suite=suite,
        git_commit=_git_commit(),
        generated_at_utc=datetime.now(UTC).isoformat(),
        python_version=platform.python_version(),
        platform=platform.platform(),
        source_dirty=source_dirty,
        source_state_digest=source_state_digest,
        data_files=files,
        config=dict(config or {}),
    )


def write_json_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_eval_run_paths(output_dir: Path, family: str, run_id: str) -> EvalRunPaths:
    family_dir = output_dir / family
    run_dir = family_dir / run_id
    return EvalRunPaths(
        run_dir=run_dir,
        json_path=run_dir / "report.json",
        md_path=run_dir / "report.md",
        latest_json_path=family_dir / "latest.json",
        latest_md_path=family_dir / "latest.md",
    )


def update_latest_eval_report(paths: EvalRunPaths) -> None:
    paths.latest_json_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(paths.json_path, paths.latest_json_path)
    shutil.copyfile(paths.md_path, paths.latest_md_path)
