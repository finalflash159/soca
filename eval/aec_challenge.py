"""Adapter for the Microsoft AEC-Challenge *real* recordings (P3.1 Pha B, Tier 1).

The ``real/`` split is the gold data for the acoustic front-end: recordings made on
10k+ real devices, each scenario shipping a paired
``<id>_<scenario>_mic.flac`` (near = what the microphone captured) and
``<id>_<scenario>_lpb.flac`` (far = the loopback / speaker reference). That pair is
exactly what ``DuplexAecSink`` consumes, so feeding ``lpb → far`` and ``mic → near``
into ``BargeInDecider`` replays barge-in detection on real echo — no synthesis.

Scenario → SoCa condition (Barański-style honesty about what each measures):

    farend_singletalk[_with_movement]   echo only, no user  → MUST NOT interrupt
                                          (measures false-interrupt / AEC quality)
    doubletalk[_with_movement]          speaker + user       → SHOULD interrupt
                                          (measures detection / missed)
    nearend_singletalk                  user only, no echo   → sanity (should fire)
    sweep                               calibration tone     → skipped

Real recordings carry no frame-precise user onset, so *latency* is not measured here
(that needs controlled synthesis; see the VN scenario builder). Tier 1 answers the
prior question: does barge-in survive real device echo without false-triggering?

English content is fine at this tier — AEC + Silero VAD are language-agnostic
(energy/echo, not words).
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

_SAMPLE_RATE = 16000
_BARGE_FRAME = 512  # 32ms @16k, aligns replay to DuplexAecSink frames

# Canonical scenario stems, longest first so the greedy suffix match is unambiguous
# (ids themselves contain '_' and '-').
_SCENARIOS_LONGEST_FIRST = (
    "farend_singletalk_with_movement",
    "doubletalk_with_movement",
    "farend_singletalk",
    "nearend_singletalk",
    "doubletalk",
    "sweep",
)

# scenario stem → (condition, expected_interrupt). ``sweep`` is intentionally absent.
_CONDITION_SPEC: dict[str, tuple[str, bool]] = {
    "farend_singletalk": ("echo_only", False),
    "farend_singletalk_with_movement": ("echo_only", False),
    "doubletalk": ("double_talk", True),
    "doubletalk_with_movement": ("double_talk", True),
    "nearend_singletalk": ("near_only", True),
}


@dataclass(frozen=True)
class AecScenario:
    """One paired mic/lpb recording with its expected barge-in outcome."""

    fileid: str
    scenario: str  # canonical stem, e.g. "doubletalk_with_movement"
    condition: str  # echo_only | double_talk | near_only
    expected_interrupt: bool
    with_movement: bool
    mic_path: Path  # near
    lpb_path: Path  # far


def _parse_stem(stem: str) -> tuple[str, str, str] | None:
    """``<id>_<scenario>_<side>`` → (fileid, scenario, side); None if unrecognised."""
    if stem.endswith("_mic"):
        side, body = "mic", stem[: -len("_mic")]
    elif stem.endswith("_lpb"):
        side, body = "lpb", stem[: -len("_lpb")]
    else:
        return None
    for scenario in _SCENARIOS_LONGEST_FIRST:
        suffix = "_" + scenario
        if body.endswith(suffix):
            return body[: -len(suffix)], scenario, side
    return None


def discover_pairs(root: Path) -> list[AecScenario]:
    """Scan ``root`` for complete mic+lpb pairs of the scored scenarios.

    Deterministic order (sorted by fileid then scenario) so downstream sampling is
    reproducible. Half-pairs (mic without lpb or vice-versa) and ``sweep`` are dropped.
    """
    sides: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
    for path in root.glob("*.flac"):
        parsed = _parse_stem(path.stem)
        if parsed is None:
            continue
        fileid, scenario, side = parsed
        if scenario not in _CONDITION_SPEC:
            continue
        sides[(fileid, scenario)][side] = path

    scenarios: list[AecScenario] = []
    for (fileid, scenario), paths in sides.items():
        if "mic" not in paths or "lpb" not in paths:
            continue
        condition, expected = _CONDITION_SPEC[scenario]
        scenarios.append(
            AecScenario(
                fileid=fileid,
                scenario=scenario,
                condition=condition,
                expected_interrupt=expected,
                with_movement=scenario.endswith("_with_movement"),
                mic_path=paths["mic"],
                lpb_path=paths["lpb"],
            )
        )
    scenarios.sort(key=lambda s: (s.fileid, s.scenario))
    return scenarios


def sample_by_condition(
    scenarios: list[AecScenario], n_per_condition: int, seed: int = 42
) -> list[AecScenario]:
    """A balanced, deterministic subset: up to ``n_per_condition`` of each condition.

    Balancing keeps false-interrupt (echo_only) and detection (double_talk) on the
    same footing regardless of how the corpus is distributed. Same seed → same items,
    mirroring the P1.1 sampling culture."""
    by_condition: dict[str, list[AecScenario]] = defaultdict(list)
    for scenario in scenarios:
        by_condition[scenario.condition].append(scenario)

    picked: list[AecScenario] = []
    for condition in sorted(by_condition):
        pool = sorted(by_condition[condition], key=lambda s: s.fileid)
        rng = random.Random(f"{seed}:{condition}")
        rng.shuffle(pool)
        picked.extend(pool[:n_per_condition])
    return picked


def _load_mono16k(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != _SAMPLE_RATE:
        raise ValueError(f"{path.name}: expected {_SAMPLE_RATE}Hz, got {sr}")
    return np.ascontiguousarray(audio, dtype=np.float32)


def load_pair(scenario: AecScenario) -> tuple[np.ndarray, np.ndarray]:
    """Load (far=lpb, near=mic), aligned to a common whole number of duplex frames.

    The two channels of a real recording can differ by a few samples; we clip both to
    the shorter one and drop the sub-frame tail so ``BargeInDecider`` sees only
    complete frames (its faithful-replay precondition)."""
    far = _load_mono16k(scenario.lpb_path)
    near = _load_mono16k(scenario.mic_path)
    n = min(len(far), len(near))
    n -= n % _BARGE_FRAME
    return far[:n], near[:n]
