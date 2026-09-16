from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def test_workflows_use_node_24_action_releases() -> None:
    workflow_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(WORKFLOWS.glob("*.yml"))
    )

    obsolete_actions = (
        "actions/checkout@v4",
        "actions/setup-node@v4",
        "actions/setup-python@v5",
        "actions/upload-artifact@v4",
        "astral-sh/setup-uv@v5",
    )
    for action in obsolete_actions:
        assert action not in workflow_text

    assert "astral-sh/setup-uv@v10.1.0" in workflow_text
