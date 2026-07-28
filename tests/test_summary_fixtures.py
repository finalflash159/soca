from __future__ import annotations

import json
from pathlib import Path

PROMPTS = Path(__file__).resolve().parents[1] / "eval" / "prompts"


def _rows(name: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in (PROMPTS / name).read_text(encoding="utf-8").splitlines() if line]


def test_summary_session_fixture_has_200_family_split_records() -> None:
    rows = _rows("summary_session_vi_v2.jsonl")
    assert len(rows) == 200
    families = {str(row["family"]): str(row["split"]) for row in rows}
    assert len(families) == 8
    assert {row["source"] for row in rows} == {"synthetic_annotated_v2"}
    assert all(row["split"] in {"train", "validation", "test"} for row in rows)
    assert len({json.dumps(row["expected"], sort_keys=True) for row in rows}) >= 150
    assert len({row["frozen_turns"][0]["user"] for row in rows}) == 200
    assert all("required_facts" in row for row in rows)


def test_summary_rolling_fixture_has_40_multi_generation_sessions() -> None:
    rows = _rows("summary_rolling_vi_v2.jsonl")
    assert len(rows) == 40
    assert all(3 <= len(row["generations"]) <= 8 for row in rows)
    assert len({json.dumps(row["expected_final"], sort_keys=True) for row in rows}) == 40
