from __future__ import annotations

import json
import operator
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from eval.release_report import GateStatus, build_report, render_markdown


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("release runner manifest must use schema_version 1")
    gates = payload.get("gates")
    if not isinstance(gates, list) or not gates:
        raise ValueError("release runner manifest requires gates")
    return payload


_CHECK_OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
    "eq": operator.eq,
    "ne": operator.ne,
    "gt": operator.gt,
    "gte": operator.ge,
    "lt": operator.lt,
    "lte": operator.le,
}


def _result_path(entry: dict[str, Any], *, repo_root: Path) -> Path | None:
    declared = entry.get("result_path")
    pattern = entry.get("result_glob")
    if declared is not None and pattern is not None:
        raise ValueError(f"{entry.get('id', '<gate>')}: result_path and result_glob are exclusive")
    if declared is not None:
        if not isinstance(declared, str) or not declared.strip():
            raise ValueError(f"{entry.get('id', '<gate>')}: result_path must be a path")
        path = Path(declared)
        if path.is_absolute():
            raise ValueError(f"{entry.get('id', '<gate>')}: result_path must be repo-relative")
        return (repo_root / path).resolve()
    if pattern is not None:
        if not isinstance(pattern, str) or not pattern.strip():
            raise ValueError(f"{entry.get('id', '<gate>')}: result_glob must be non-empty")
        matches = sorted(
            (candidate for candidate in repo_root.glob(pattern) if candidate.is_file()),
            key=lambda candidate: candidate.stat().st_mtime_ns,
            reverse=True,
        )
        return matches[0].resolve() if matches else None
    return None


def _read_path(payload: Any, path: list[Any], *, gate_id: str) -> Any:
    current = payload
    for part in path:
        if isinstance(current, dict) and isinstance(part, str) and part in current:
            current = current[part]
        elif isinstance(current, list) and isinstance(part, int) and not isinstance(part, bool):
            try:
                current = current[part]
            except IndexError as exc:
                raise ValueError(f"{gate_id}: result path index is out of range: {path!r}") from exc
        else:
            raise ValueError(f"{gate_id}: result path does not exist: {path!r}")
    return current


def _check_result(payload: Any, entry: dict[str, Any]) -> list[dict[str, Any]]:
    gate_id = str(entry["id"])
    checks = entry.get("checks", [])
    if not isinstance(checks, list):
        raise ValueError(f"{gate_id}: checks must be a list")
    failures: list[dict[str, Any]] = []
    for raw in checks:
        if not isinstance(raw, dict):
            raise ValueError(f"{gate_id}: every check must be an object")
        path = raw.get("path")
        operator_name = raw.get("operator")
        if (
            not isinstance(path, list)
            or not path
            or any(not isinstance(part, (str, int)) or isinstance(part, bool) for part in path)
        ):
            raise ValueError(f"{gate_id}: check path must be a non-empty string/int list")
        if not isinstance(operator_name, str) or operator_name not in _CHECK_OPERATORS:
            raise ValueError(f"{gate_id}: unsupported check operator: {operator_name!r}")
        if "value" not in raw:
            raise ValueError(f"{gate_id}: check value is required")
        try:
            actual = _read_path(payload, path, gate_id=gate_id)
            passed = bool(_CHECK_OPERATORS[operator_name](actual, raw["value"]))
        except (TypeError, ValueError) as exc:
            failures.append({"path": path, "operator": operator_name, "error": str(exc)})
            continue
        if not passed:
            failures.append(
                {
                    "path": path,
                    "operator": operator_name,
                    "expected": raw["value"],
                    "actual": actual,
                }
            )
    return failures


def run_manifest(
    manifest_path: Path,
    *,
    repo_root: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], Path]:
    manifest = _load_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_entries: list[dict[str, Any]] = []
    for entry in manifest["gates"]:
        if not isinstance(entry, dict):
            raise ValueError("release runner gate entries must be objects")
        gate_id = entry.get("id")
        command = entry.get("command")
        if not isinstance(gate_id, str) or not gate_id.strip():
            raise ValueError("release runner gate id must be non-empty")
        if not isinstance(command, list) or any(not isinstance(item, str) for item in command):
            raise ValueError(f"{gate_id}: command must be a list of strings")
        timeout = entry.get("timeout_seconds")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError(f"{gate_id}: timeout_seconds must be positive")
        log_path = output_dir / f"{gate_id}.log"
        started = time.monotonic()
        status = GateStatus.PASS
        reason = "command completed successfully"
        details: dict[str, Any] = {}
        evidence_paths = [log_path]
        result_path: Path | None = None
        try:
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=repo_root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=float(timeout),
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
            details["return_code"] = completed.returncode
            if completed.returncode != 0:
                status = GateStatus.FAIL
                reason = "command returned a non-zero exit code"
            result_path = _result_path(entry, repo_root=repo_root)
            if result_path is None and ("result_path" in entry or "result_glob" in entry):
                status = GateStatus.FAIL
                reason = "declared result evidence is missing"
                details["result_error"] = "no result file matched the declaration"
            elif result_path is not None:
                evidence_paths.append(result_path)
                if result_path.is_file() and entry.get("checks", []):
                    try:
                        result_payload = json.loads(result_path.read_text(encoding="utf-8"))
                        failures = _check_result(result_payload, entry)
                    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                        failures = [{"error": str(exc)}]
                    if failures:
                        status = GateStatus.FAIL
                        reason = "declared evidence checks failed"
                        details["check_failures"] = failures
        except subprocess.TimeoutExpired as exc:
            status = GateStatus.FAIL
            reason = "command exceeded its declared timeout"
            details["timeout_seconds"] = timeout
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\nTIMEOUT: {exc}\n")
        except OSError as exc:
            status = GateStatus.BLOCKED
            reason = "command could not be started"
            details["error_type"] = type(exc).__name__
            details["error"] = str(exc)
            log_path.write_text(f"START ERROR: {type(exc).__name__}: {exc}\n", encoding="utf-8")
        details["duration_ms"] = round((time.monotonic() - started) * 1000, 3)
        result_entries.append(
            {
                "id": gate_id,
                "status": status.value,
                "required": entry.get("required") is True,
                "reason": reason,
                "evidence": [str(path) for path in evidence_paths if path.is_file()],
                "command": command,
                "details": details,
            }
        )

    result_manifest = output_dir / "results.json"
    result_manifest.write_text(
        json.dumps(
            {"schema_version": 1, "gates": result_entries},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = build_report(
        manifest=result_manifest,
        repo_root=repo_root,
        suite=str(manifest.get("suite", "release-gates")),
    )
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    return report, report_path
