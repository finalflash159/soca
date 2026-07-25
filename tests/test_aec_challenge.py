"""Tests for the AEC-Challenge real-pair adapter (P3.1 Pha B, Tier 1).

Writes tiny synthetic flac pairs into tmp_path — no 40GB corpus, no network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from eval.aec_challenge import (
    _parse_stem,
    discover_pairs,
    load_pair,
    sample_by_condition,
)

_SR = 16000
_FRAME = 512


def _write(path: Path, n_samples: int, value: float = 0.0) -> None:
    sf.write(path, np.full(n_samples, value, dtype=np.float32), _SR)


def _pair(root: Path, fileid: str, scenario: str, mic_len: int, lpb_len: int) -> None:
    _write(root / f"{fileid}_{scenario}_mic.flac", mic_len)
    _write(root / f"{fileid}_{scenario}_lpb.flac", lpb_len)


def test_parse_stem_handles_underscored_ids_and_movement() -> None:
    assert _parse_stem("abc_doubletalk_mic") == ("abc", "doubletalk", "mic")
    assert _parse_stem("a_b_c_farend_singletalk_with_movement_lpb") == (
        "a_b_c",
        "farend_singletalk_with_movement",
        "lpb",
    )
    # movement variant must win over the plain stem (longest-first match).
    assert _parse_stem("x_doubletalk_with_movement_mic") == (
        "x",
        "doubletalk_with_movement",
        "mic",
    )


def test_parse_stem_rejects_unknown_or_sideless() -> None:
    assert _parse_stem("x_not_a_scenario_mic") is None
    assert _parse_stem("x_doubletalk_other") is None  # bad side
    assert _parse_stem("no_side_here") is None


def test_discover_pairs_labels_conditions_and_drops_incomplete(tmp_path: Path) -> None:
    _pair(tmp_path, "id1", "doubletalk", 3 * _FRAME, 3 * _FRAME)
    _pair(tmp_path, "id2", "farend_singletalk", 3 * _FRAME, 3 * _FRAME)
    _pair(tmp_path, "id3", "farend_singletalk_with_movement", 3 * _FRAME, 3 * _FRAME)
    _pair(tmp_path, "id4", "sweep", 3 * _FRAME, 3 * _FRAME)  # dropped: not scored
    _write(tmp_path / "id5_doubletalk_mic.flac", 3 * _FRAME)  # half-pair: dropped

    pairs = discover_pairs(tmp_path)
    by_id = {p.fileid: p for p in pairs}

    assert set(by_id) == {"id1", "id2", "id3"}  # sweep + half-pair gone
    assert by_id["id1"].condition == "double_talk" and by_id["id1"].expected_interrupt
    assert by_id["id2"].condition == "echo_only" and not by_id["id2"].expected_interrupt
    assert by_id["id3"].with_movement is True
    assert by_id["id1"].with_movement is False


def test_sample_by_condition_is_balanced_and_deterministic(tmp_path: Path) -> None:
    for i in range(5):
        _pair(tmp_path, f"dt{i}", "doubletalk", 2 * _FRAME, 2 * _FRAME)
        _pair(tmp_path, f"fe{i}", "farend_singletalk", 2 * _FRAME, 2 * _FRAME)
    pairs = discover_pairs(tmp_path)

    a = sample_by_condition(pairs, n_per_condition=2, seed=42)
    b = sample_by_condition(pairs, n_per_condition=2, seed=42)

    conditions = sorted(p.condition for p in a)
    assert conditions == ["double_talk", "double_talk", "echo_only", "echo_only"]
    assert [p.fileid for p in a] == [p.fileid for p in b]  # deterministic


def test_load_pair_aligns_to_whole_frames(tmp_path: Path) -> None:
    # mic and lpb differ, and neither is a frame multiple → both clip to 3 frames.
    _pair(tmp_path, "id1", "doubletalk", mic_len=3 * _FRAME + 10, lpb_len=3 * _FRAME + 60)
    scenario = discover_pairs(tmp_path)[0]

    far, near = load_pair(scenario)

    assert len(far) == len(near) == 3 * _FRAME
    assert len(far) % _FRAME == 0


def test_load_pair_rejects_wrong_sample_rate(tmp_path: Path) -> None:
    sf.write(tmp_path / "id1_doubletalk_mic.flac", np.zeros(512, np.float32), 8000)
    sf.write(tmp_path / "id1_doubletalk_lpb.flac", np.zeros(512, np.float32), 8000)
    scenario = discover_pairs(tmp_path)[0]

    with pytest.raises(ValueError, match="expected 16000Hz"):
        load_pair(scenario)
