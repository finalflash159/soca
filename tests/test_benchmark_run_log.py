from __future__ import annotations

import json
import sys

from eval.benchmark_run import run_logged_command


def test_logged_benchmark_captures_output_and_provenance(tmp_path) -> None:
    run_dir = tmp_path / "run"

    exit_code = run_logged_command(
        (sys.executable, "-c", "print('measured')"),
        run_dir=run_dir,
        family="retrieval",
    )

    assert exit_code == 0
    assert "measured" in (run_dir / "run.log").read_text(encoding="utf-8")
    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["family"] == "retrieval"
    assert payload["command"][-1] == "print('measured')"
    assert payload["exit_code"] == 0
    assert payload["started_at_utc"]
    assert payload["finished_at_utc"]
    assert payload["source"]["tracked_tree_sha256"]
    assert payload["source"]["working_diff_sha256"]
