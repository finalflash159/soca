from __future__ import annotations

import pytest

from eval.calibrate_router import _fit_margin
from soca.core.calibration import CalibrationArtifact


def _artifact() -> CalibrationArtifact:
    return CalibrationArtifact(
        version=1,
        encoder_id="fake:router",
        aggregation="max",
        route_thresholds={
            "direct_tool": 0.8,
            "retrieval_request": 0.7,
            "smalltalk": 0.7,
            "out_of_scope": 0.9,
            "unresolved": 0.9,
        },
        route_margin=0.05,
        source_thresholds={"knowledge": 0.6, "memory": 0.6},
        source_margin=0.05,
        metadata={"final_test_sealed": True},
    )


def test_calibration_artifact_abstains_below_route_margin() -> None:
    artifact = _artifact()
    route, runner_up, margin = artifact.route(
        {
            "direct_tool": 0.82,
            "retrieval_request": 0.80,
            "smalltalk": 0.2,
            "out_of_scope": 0.1,
            "unresolved": 0.1,
        }
    )
    assert route == "unresolved"
    assert runner_up == "retrieval_request"
    assert margin == pytest.approx(0.02)


def test_calibration_artifact_keeps_both_source_profile() -> None:
    profile, selected = _artifact().select_sources({"knowledge": 0.8, "memory": 0.79})
    assert profile == "both"
    assert selected == ("knowledge", "memory")


def test_calibration_reports_unsupported_count_and_oos_denominator() -> None:
    rows = [
        {
            "scores": {
                "direct_tool": 0.90,
                "retrieval_request": 0.10,
                "smalltalk": 0.10,
                "out_of_scope": 0.10,
                "unresolved": 0.10,
            },
            "disposition": "out_of_scope",
        },
        {
            "scores": {
                "direct_tool": 0.80,
                "retrieval_request": 0.10,
                "smalltalk": 0.10,
                "out_of_scope": 0.10,
                "unresolved": 0.10,
            },
            "disposition": "out_of_scope",
        },
        {
            "scores": {
                "direct_tool": 0.10,
                "retrieval_request": 0.10,
                "smalltalk": 0.90,
                "out_of_scope": 0.10,
                "unresolved": 0.10,
            },
            "disposition": "smalltalk",
        },
    ]
    thresholds = {name: 0.0 for name in rows[0]["scores"]}

    _, metrics = _fit_margin(rows, thresholds=thresholds, source=False)

    assert metrics["unsupported_count"] == 2
    assert metrics["unsupported_total"] == 2
    assert metrics["unsupported_rate"] == 1.0
