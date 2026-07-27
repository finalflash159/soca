from __future__ import annotations

import pytest

from eval.eval_voice_ttfa import compare_reports


def _report(values: list[float]) -> dict:
    return {
        "profiles": [
            {
                "profile": "baseline",
                "samples": [
                    {"sample_id": str(index), "ttfa_ms": value}
                    for index, value in enumerate(values)
                ],
            }
        ]
    }


def test_compare_reports_requires_pairs_and_checks_budget() -> None:
    result = compare_reports(_report([100, 110, 120]), _report([105, 115, 125]), profile="baseline")
    assert result["sample_count"] == 3
    assert result["within_budget"] is True


def test_compare_reports_rejects_too_few_pairs() -> None:
    with pytest.raises(ValueError, match="at least three"):
        compare_reports(_report([100, 110]), _report([105, 115]), profile="baseline")
