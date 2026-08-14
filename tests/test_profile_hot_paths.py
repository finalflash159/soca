from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.profile_hot_paths import (
    classify_speedscope,
    native_sampling_supported,
    parse_py_spy_version,
)


def _write_profile(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "$schema": "https://www.speedscope.app/file-format-schema.json",
                "shared": {
                    "frames": [
                        {"name": "root", "file": "/tmp/workload.py"},
                        {"name": "search", "file": "/tmp/retrieval.py"},
                        {"name": "onnxruntime::Run", "file": "libonnxruntime.dylib"},
                    ]
                },
                "profiles": [
                    {
                        "type": "sampled",
                        "samples": [[0, 1], [0, 2], [0, 2]],
                        "weights": [1, 2, 1],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_speedscope_classifies_exclusive_leaf_frame_with_weights(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    _write_profile(profile)

    result = classify_speedscope(profile, native_available=True)

    assert result["method"] == "exclusive_leaf_frame"
    assert result["weighted_samples"] == 4.0
    assert result["python_percent"] == 25.0
    assert result["native_percent"] == 75.0


def test_speedscope_refuses_to_infer_native_split_when_unsupported(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    _write_profile(profile)

    result = classify_speedscope(profile, native_available=False)

    assert result["status"] == "blocked"
    assert result["python_percent"] is None
    assert result["native_percent"] is None
    assert result["reason"] == "native_sampling_unsupported"


def test_py_spy_version_parser_is_strict() -> None:
    assert parse_py_spy_version("py-spy 0.4.2\n") == "0.4.2"
    with pytest.raises(ValueError, match="unexpected"):
        parse_py_spy_version("something else")


def test_native_support_matrix_does_not_treat_apple_arm_as_sbc_evidence() -> None:
    assert native_sampling_supported(system="Darwin", machine="arm64") is False
    assert native_sampling_supported(system="Linux", machine="aarch64") is True
