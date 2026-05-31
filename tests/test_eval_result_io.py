from __future__ import annotations

from eval.result_io import make_eval_run_paths, update_latest_eval_report


def test_make_eval_run_paths_groups_report_by_family_and_run_id(tmp_path):
    paths = make_eval_run_paths(tmp_path, "tts_bakeoff", "20260601_010203")

    assert paths.run_dir == tmp_path / "tts_bakeoff" / "20260601_010203"
    assert paths.json_path == paths.run_dir / "report.json"
    assert paths.md_path == paths.run_dir / "report.md"
    assert paths.latest_json_path == tmp_path / "tts_bakeoff" / "latest.json"
    assert paths.latest_md_path == tmp_path / "tts_bakeoff" / "latest.md"


def test_update_latest_eval_report_copies_current_report(tmp_path):
    paths = make_eval_run_paths(tmp_path, "llm_bakeoff", "20260601_010203")
    paths.run_dir.mkdir(parents=True)
    paths.json_path.write_text('{"ok": true}', encoding="utf-8")
    paths.md_path.write_text("# Report\n", encoding="utf-8")

    update_latest_eval_report(paths)

    assert paths.latest_json_path.read_text(encoding="utf-8") == '{"ok": true}'
    assert paths.latest_md_path.read_text(encoding="utf-8") == "# Report\n"
