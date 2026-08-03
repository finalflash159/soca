"""Build an explicit, provenance-carrying release-gate report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class GateStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    BLOCKED = "blocked"
    UNSUPPORTED = "unsupported"


_TERMINAL_STATUSES = frozenset(GateStatus)


@dataclass(frozen=True)
class EvidenceRef:
    path: str
    sha256: str
    bytes: int
    private: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "private": self.private,
        }


@dataclass(frozen=True)
class GateResult:
    gate_id: str
    status: GateStatus
    required: bool
    reason: str
    evidence: tuple[EvidenceRef, ...] = ()
    command: tuple[str, ...] = ()
    details: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.gate_id.strip():
            raise ValueError("gate_id must not be empty")
        if not self.reason.strip():
            raise ValueError("gate reason must not be empty")
        if self.status is GateStatus.PASS and not self.evidence:
            raise ValueError(f"passing gate {self.gate_id!r} requires evidence")
        if self.status is GateStatus.FAIL and not self.evidence:
            raise ValueError(f"failed gate {self.gate_id!r} requires evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.gate_id,
            "status": self.status.value,
            "required": self.required,
            "reason": self.reason,
            "evidence": [item.to_dict() for item in self.evidence],
            "command": list(self.command),
            "details": dict(self.details or {}),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_path(path: Path, repo_root: Path) -> tuple[str, bool]:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix(), False
    except ValueError:
        return f"<local-only>/{resolved.name}", True


def evidence_ref(path: Path, *, repo_root: Path) -> EvidenceRef:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"release evidence is missing: {resolved}")
    safe, private = _safe_path(resolved, repo_root)
    return EvidenceRef(
        path=safe,
        sha256=sha256_file(resolved),
        bytes=resolved.stat().st_size,
        private=private,
    )


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _repo_dirty(repo_root: Path) -> bool:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return True
    return result.returncode != 0 or bool(result.stdout.strip())


def _require_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def load_manifest(path: Path, *, repo_root: Path) -> tuple[GateResult, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("release gate manifest must use schema_version 1")
    raw_gates = payload.get("gates")
    if not isinstance(raw_gates, list) or not raw_gates:
        raise ValueError("release gate manifest requires a non-empty gates list")

    results: list[GateResult] = []
    seen: set[str] = set()
    for raw in raw_gates:
        if not isinstance(raw, dict):
            raise ValueError("release gate entries must be objects")
        gate_id = _require_string(raw.get("id"), field="gate id")
        if gate_id in seen:
            raise ValueError(f"duplicate release gate: {gate_id}")
        seen.add(gate_id)
        try:
            status = GateStatus(_require_string(raw.get("status"), field=f"{gate_id}.status"))
        except ValueError as exc:
            raise ValueError(f"{gate_id}: invalid gate status") from exc
        evidence_paths = raw.get("evidence", [])
        if not isinstance(evidence_paths, list) or any(not isinstance(item, str) for item in evidence_paths):
            raise ValueError(f"{gate_id}.evidence must be a list of paths")
        evidence = tuple(
            evidence_ref(Path(item), repo_root=repo_root) for item in evidence_paths
        )
        command = raw.get("command", [])
        if not isinstance(command, list) or any(not isinstance(item, str) for item in command):
            raise ValueError(f"{gate_id}.command must be a list of strings")
        details = raw.get("details", {})
        if not isinstance(details, dict):
            raise ValueError(f"{gate_id}.details must be an object")
        results.append(
            GateResult(
                gate_id=gate_id,
                status=status,
                required=raw.get("required") is True,
                reason=_require_string(raw.get("reason"), field=f"{gate_id}.reason"),
                evidence=evidence,
                command=tuple(command),
                details=details,
            )
        )
    return tuple(results)


def build_report(
    *,
    manifest: Path,
    repo_root: Path,
    suite: str,
) -> dict[str, Any]:
    gates = load_manifest(manifest, repo_root=repo_root)
    required_failures = [
        gate.gate_id
        for gate in gates
        if gate.required and gate.status is not GateStatus.PASS
    ]
    report = {
        "schema_version": "soca-release-gates-v1",
        "suite": suite,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "commit": _git_commit(repo_root),
            "repo_dirty": _repo_dirty(repo_root),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "gates": [gate.to_dict() for gate in gates],
        "decision": {
            "required_gate_count": sum(gate.required for gate in gates),
            "required_failures": required_failures,
            "status": "pass" if not required_failures else "blocked",
            "no_silent_skip": True,
        },
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        f"# {report['suite']}",
        "",
        f"- Decision: **{decision['status']}**",
        f"- Commit: `{report['source']['commit']}`",
        f"- Recorded: `{report['recorded_at_utc']}`",
        "",
        "| Gate | Required | Status | Reason | Evidence |",
        "|---|---:|---|---|---|",
    ]
    for gate in report["gates"]:
        evidence = ", ".join(item["path"] for item in gate["evidence"]) or "—"
        lines.append(
            f"| `{gate['id']}` | {'yes' if gate['required'] else 'no'} | "
            f"`{gate['status']}` | {gate['reason']} | {evidence} |"
        )
    lines.extend(
        [
            "",
            "A non-pass required gate blocks release. Missing evidence is never treated as a pass.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--suite", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(
        manifest=args.manifest,
        repo_root=args.repo_root,
        suite=args.suite,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    return 0 if report["decision"]["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
