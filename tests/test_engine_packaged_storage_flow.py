from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

from soca.memory import SessionCheckpointStore, WorkingMemory


def _run_engine(env: dict[str, str]) -> list[dict[str, object]]:
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "from soca.cli import main; main()",
            "engine",
            "--no-model",
            "--session-persistence",
            "local_resumable",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        input='{"cmd":"sessions_list","limit":10}\n{"cmd":"quit"}\n',
        capture_output=True,
        text=True,
        check=True,
    )
    return [json.loads(line) for line in process.stdout.splitlines() if line.strip()]


def test_engine_migrates_legacy_checkpoints_into_the_packaged_xdg_root(tmp_path: Path) -> None:
    legacy_root = tmp_path / "legacy-sessions"
    working = WorkingMemory(thread_id="desktop-legacy")
    turn = working.begin_turn("Mở lại phiên sau khi cập nhật")
    working.finish_turn(turn.sequence, "Context đã được lưu")
    SessionCheckpointStore(legacy_root).save(working)

    app_data = tmp_path / "app-data"
    environment = {
        **os.environ,
        "XDG_CONFIG_HOME": str(app_data / "config"),
        "XDG_DATA_HOME": str(app_data / "data"),
        "XDG_STATE_HOME": str(app_data / "state"),
        "SOCA_VAULT": str(app_data / "vault"),
        "SOCA_LEGACY_SESSION_ROOT": str(legacy_root),
    }

    first = _run_engine(environment)
    page = next(frame for frame in first if frame["event"] == "sessions_page")
    sessions = page["sessions"]
    assert isinstance(sessions, list)
    assert len(sessions) == 2  # imported checkpoint plus the fresh active aggregate
    assert any(session["checkpoint_only"] is True for session in sessions)

    database = app_data / "state" / "soca" / "sessions" / "sessions.sqlite3"
    assert database.is_file()
    if os.name != "nt":
        assert stat.S_IMODE(database.stat().st_mode) == 0o600
    backups = sorted(
        (app_data / "state" / "soca" / "sessions" / "legacy-backups").glob("*/manifest.json")
    )
    assert len(backups) == 1

    second = _run_engine(environment)
    second_page = next(frame for frame in second if frame["event"] == "sessions_page")
    second_sessions = second_page["sessions"]
    assert isinstance(second_sessions, list)
    assert len(second_sessions) == 3  # migration is idempotent; only the new active aggregate is added
