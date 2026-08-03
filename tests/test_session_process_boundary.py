import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_session_checkpoint_survives_process_boundary(tmp_path: Path) -> None:
    script = """
from pathlib import Path
from soca.memory import SessionCheckpointStore, SessionMemory
import sys

root = Path(sys.argv[1])
store = SessionCheckpointStore(root)
memory = SessionMemory(
    thread_id="process-boundary",
    persistence="local_resumable",
    checkpoint_store=store,
    summary_enabled=False,
)
memory.append("user", "quyết định giữ checkpoint")
memory.append("assistant", "đã lưu")
memory.close()
"""
    subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "sessions")],
        check=True,
        cwd=Path.cwd(),
    )
    reader = subprocess.run(
        [
            sys.executable,
            "-c",
            "from soca.memory import SessionCheckpointStore, SessionMemory; "
            "import sys; "
            "m=SessionMemory(thread_id='process-boundary', persistence='local_resumable', "
            "checkpoint_store=SessionCheckpointStore(sys.argv[1]), resume=True, summary_enabled=False); "
            "print(m.render())",
            str(tmp_path / "sessions"),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path.cwd(),
    )
    assert "quyết định giữ checkpoint" in reader.stdout
    expected_name = hashlib.sha256(b"process-boundary").hexdigest() + ".json"
    checkpoint = tmp_path / "sessions" / expected_name
    assert checkpoint.exists()
    files = sorted((tmp_path / "sessions").glob("*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text(encoding="utf-8"))["schema_version"] == 1
