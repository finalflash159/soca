from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from soca import cli


class FakeIndexCoordinator:
    def inspect(self) -> dict[str, object]:
        return {
            "status": {"dense_state": "ready"},
            "pointers": [{"active_generation_id": "generation-2"}],
            "generations": [{"id": "generation-2", "state": "READY"}],
            "jobs": [],
        }

    def sync_sparse(self, *, verify_content: bool) -> object:
        assert verify_content is True
        return SimpleNamespace(
            revision=3,
            changed=True,
            index=SimpleNamespace(records=(object(), object()), chunks=(object(),)),
        )

    def rollback(self) -> str:
        return "generation-1"


def test_index_operator_commands_emit_machine_readable_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    coordinator = FakeIndexCoordinator()
    monkeypatch.setattr(cli, "_index_context", lambda *args, **kwargs: coordinator)
    runner = CliRunner()

    inspected = runner.invoke(
        cli.main,
        ["knowledge", "index", "inspect", "--vault", str(tmp_path)],
    )
    migrated = runner.invoke(
        cli.main,
        ["knowledge", "index", "migrate", "--vault", str(tmp_path)],
    )
    rolled_back = runner.invoke(
        cli.main,
        ["knowledge", "index", "rollback", "--vault", str(tmp_path)],
    )

    assert inspected.exit_code == 0
    assert json.loads(inspected.output)["status"]["dense_state"] == "ready"
    assert migrated.exit_code == 0
    assert json.loads(migrated.output) == {
        "changed": True,
        "chunks": 1,
        "documents": 2,
        "revision": 3,
    }
    assert rolled_back.exit_code == 0
    assert json.loads(rolled_back.output) == {"active_generation": "generation-1"}
