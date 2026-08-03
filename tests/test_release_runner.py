import json
import sys
from pathlib import Path

from eval.release_runner import run_manifest


def _manifest(path: Path, command: list[str], **gate_fields: object) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "suite": "runner-test",
                "gates": [
                    {
                        "id": "command",
                        "required": True,
                        "timeout_seconds": 5,
                        "command": command,
                        **gate_fields,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_runner_records_success_and_raw_log(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, [sys.executable, "-c", "print('ok')"])

    report, report_path = run_manifest(
        manifest,
        repo_root=Path.cwd(),
        output_dir=tmp_path / "run",
    )

    assert report_path.is_file()
    assert report["decision"]["status"] == "pass"
    assert "ok" in (tmp_path / "run" / "command.log").read_text(encoding="utf-8")


def test_runner_marks_nonzero_command_as_required_failure(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _manifest(manifest, [sys.executable, "-c", "raise SystemExit(3)"])

    report, _ = run_manifest(manifest, repo_root=Path.cwd(), output_dir=tmp_path / "run")

    assert report["decision"]["status"] == "blocked"
    assert report["decision"]["required_failures"] == ["command"]
    assert report["gates"][0]["details"]["return_code"] == 3


def test_runner_requires_declared_result_checks(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    manifest = tmp_path / "manifest.json"
    _manifest(
        manifest,
        [
            sys.executable,
            "-c",
            "import json, pathlib; pathlib.Path('result.json').write_text(json.dumps({'ok': True, 'score': 0.4}))",
        ],
        result_path="result.json",
        checks=[
            {"path": ["ok"], "operator": "eq", "value": True},
            {"path": ["score"], "operator": "gte", "value": 0.5},
        ],
    )

    report, _ = run_manifest(manifest, repo_root=tmp_path, output_dir=tmp_path / "run")

    assert result.is_file()
    assert report["decision"]["status"] == "blocked"
    assert report["gates"][0]["status"] == "fail"
    assert report["gates"][0]["details"]["check_failures"][0]["path"] == ["score"]


def test_runner_records_checked_result_as_evidence(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _manifest(
        manifest,
        [
            sys.executable,
            "-c",
            "import json, pathlib; pathlib.Path('result.json').write_text(json.dumps({'ok': True}))",
        ],
        result_path="result.json",
        checks=[{"path": ["ok"], "operator": "eq", "value": True}],
    )

    report, _ = run_manifest(manifest, repo_root=tmp_path, output_dir=tmp_path / "run")

    assert report["decision"]["status"] == "pass"
    assert {item["path"] for item in report["gates"][0]["evidence"]} == {
        "run/command.log",
        "result.json",
    }
