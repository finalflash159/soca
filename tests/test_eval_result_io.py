from __future__ import annotations

from pathlib import Path

from eval.result_io import (
    _git_source_state,
    make_eval_artifact_metadata,
    make_eval_run_paths,
    update_latest_eval_report,
)


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


def test_eval_metadata_includes_hardware(tmp_path):
    data_file = tmp_path / "cases.jsonl"
    data_file.write_text("{}\n", encoding="utf-8")

    artifact = make_eval_artifact_metadata(
        suite="contract",
        data_files=(data_file,),
    ).to_dict()

    assert artifact["environment"]["hardware"]["cpu_count"]
    assert "memory_bytes" in artifact["environment"]["hardware"]


def test_git_source_state_can_explicitly_ignore_preexisting_untracked_file(
    tmp_path,
    monkeypatch,
):
    ignored = tmp_path / "ignored.txt"
    ignored.write_text("user data", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    def fake_check_output(command, **_kwargs):
        if command[:3] == ["git", "diff", "HEAD"]:
            return b""
        if command[:3] == ["git", "ls-files", "--others"]:
            return b"ignored.txt\0"
        raise AssertionError(command)

    monkeypatch.setattr("eval.result_io.subprocess.check_output", fake_check_output)

    dirty, digest = _git_source_state((Path("ignored.txt"),))

    assert dirty is False
    assert digest
