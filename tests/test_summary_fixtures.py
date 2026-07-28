from __future__ import annotations

import json
from pathlib import Path

PROMPTS = Path(__file__).resolve().parents[1] / "eval" / "prompts"


def _rows(name: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in (PROMPTS / name).read_text(encoding="utf-8").splitlines() if line]


def test_summary_session_fixture_has_200_family_split_records() -> None:
    rows = _rows("summary_session_vi_v1.jsonl")
    assert len(rows) == 200
    families = {str(row["family"]): str(row["split"]) for row in rows}
    assert len(families) == 8
    assert {row["source"] for row in rows} == {"synthetic_template_v1"}
    assert all(row["split"] in {"train", "validation", "test"} for row in rows)


def test_summary_rolling_fixture_has_40_multi_generation_sessions() -> None:
    rows = _rows("summary_rolling_vi_v1.jsonl")
    assert len(rows) == 40
    assert all(len(row["generations"]) >= 2 for row in rows)
