"""Exercise a frozen desktop sidecar outside the checkout it was built from."""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from soca.memory import SessionCheckpointStore, WorkingMemory


def _run_engine(sidecar: Path, *, environment: dict[str, str], cwd: Path) -> list[dict[str, Any]]:
    process = subprocess.run(
        [
            str(sidecar),
            "engine",
            "--no-model",
            "--session-persistence",
            "local_resumable",
        ],
        cwd=cwd,
        env=environment,
        input='{"cmd":"sessions_list","limit":10}\n{"cmd":"quit"}\n',
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"frozen sidecar exited {process.returncode}\nstdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
    try:
        return [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"frozen sidecar emitted invalid NDJSON: {process.stdout}") from exc


def _legacy_checkpoint(root: Path) -> None:
    memory = WorkingMemory(thread_id="frozen-sidecar-legacy")
    turn = memory.begin_turn("Khôi phục phiên từ runtime cũ")
    memory.finish_turn(turn.sequence, "Context cũ còn nguyên")
    SessionCheckpointStore(root).save(memory)


def verify(sidecar: Path) -> None:
    if not sidecar.is_file():
        raise RuntimeError(f"sidecar does not exist: {sidecar}")

    with tempfile.TemporaryDirectory(prefix="soca-frozen-sidecar-") as temporary:
        root = Path(temporary)
        legacy = root / "legacy"
        _legacy_checkpoint(legacy)
        app_data = root / "app-data"
        isolated_cwd = root / "outside-checkout"
        isolated_cwd.mkdir()
        environment = {
            **os.environ,
            "XDG_CONFIG_HOME": str(app_data / "config"),
            "XDG_DATA_HOME": str(app_data / "data"),
            "XDG_STATE_HOME": str(app_data / "state"),
            "SOCA_VAULT": str(app_data / "vault"),
            "SOCA_LEGACY_SESSION_ROOT": str(legacy),
        }
        # A frozen executable must not inherit an editable-install route.
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        environment.pop("VIRTUAL_ENV", None)

        first = _run_engine(sidecar, environment=environment, cwd=isolated_cwd)
        hello_frames = [frame for frame in first if frame.get("event") == "hello"]
        if not hello_frames:
            raise RuntimeError("frozen sidecar did not complete the protocol hello")
        if len(hello_frames) != 1:
            raise RuntimeError("frozen sidecar emitted more than one protocol hello")
        page = next((frame for frame in first if frame.get("event") == "sessions_page"), None)
        if page is None or not any(item.get("checkpoint_only") for item in page.get("sessions", [])):
            raise RuntimeError("frozen sidecar did not migrate the legacy session")

        database = app_data / "state" / "soca" / "sessions" / "sessions.sqlite3"
        if not database.is_file():
            raise RuntimeError("frozen sidecar did not create its XDG session database")
        if os.name != "nt" and stat.S_IMODE(database.stat().st_mode) != 0o600:
            raise RuntimeError("frozen sidecar session database is not private")
        manifests = list((app_data / "state" / "soca" / "sessions" / "legacy-backups").glob("*/manifest.json"))
        if len(manifests) != 1:
            raise RuntimeError("frozen sidecar did not preserve exactly one migration backup manifest")

        second = _run_engine(sidecar, environment=environment, cwd=isolated_cwd)
        second_page = next((frame for frame in second if frame.get("event") == "sessions_page"), None)
        if second_page is None or sum(item.get("checkpoint_only") is True for item in second_page.get("sessions", [])) != 1:
            raise RuntimeError("frozen sidecar migration was not idempotent")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", required=True, type=Path)
    args = parser.parse_args()
    verify(args.sidecar.expanduser().resolve())
    print("frozen sidecar storage flow passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
