import json
import subprocess
from pathlib import Path

import pytest

from eval.release_report import GateStatus, build_report, load_manifest, render_markdown


def _write_manifest(path: Path, *, evidence: str, status: str = "pass") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "gates": [
                    {
                        "id": "checkpoint",
                        "status": status,
                        "required": True,
                        "reason": "process-boundary evidence recorded",
                        "evidence": [evidence],
                        "command": ["uv", "run", "pytest", "-q"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_release_report_hashes_evidence_and_passes_required_gate(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"status":"pass"}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, evidence=str(evidence))

    report = build_report(manifest=manifest, repo_root=Path.cwd(), suite="release")

    assert report["decision"]["status"] == "pass"
    assert report["gates"][0]["status"] == GateStatus.PASS
    assert report["gates"][0]["evidence"][0]["sha256"]


def test_release_report_blocks_required_non_pass(tmp_path: Path) -> None:
    evidence = tmp_path / "blocked.json"
    evidence.write_text('{"status":"blocked"}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, evidence=str(evidence), status="blocked")

    report = build_report(manifest=manifest, repo_root=Path.cwd(), suite="release")

    assert report["decision"] == {
        "required_gate_count": 1,
        "required_failures": ["checkpoint"],
        "status": "blocked",
        "no_silent_skip": True,
    }
    assert "blocked" in render_markdown(report)


def test_passing_gate_without_evidence_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "gates": [
                    {
                        "id": "missing",
                        "status": "pass",
                        "required": True,
                        "reason": "claimed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires evidence"):
        load_manifest(manifest, repo_root=Path.cwd())


def test_release_report_marks_staged_and_untracked_files_dirty(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("initial\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "initial"], check=True)

    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"status":"pass"}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, evidence=str(evidence))
    tracked.write_text("staged change\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    (tmp_path / "untracked.txt").write_text("untracked\n", encoding="utf-8")

    report = build_report(manifest=manifest, repo_root=tmp_path, suite="release")

    assert report["source"]["repo_dirty"] is True
