"""Conversation-eval metrics (P3.1 Pha C). Tier 1: barge-in on real echo.

Pure functions over decision outcomes — no audio, no model — so they test fast and
the same report shape serves synthetic and real runs alike (mirrors the P1.1
``robustness_metrics`` split).

Tier 1 answers the acoustic-front-end question with AEC-Challenge *real* pairs:

    echo_only  (speaker only, no user)  → any interrupt is a FALSE interrupt
    double_talk(speaker + user)         → an interrupt is a correct DETECTION;
                                          silence is a MISS

Both map onto Full-Duplex-Bench vocabulary: ``false_interrupt_rate`` is the
device-echo analogue of FDB's Takeover Rate (cutting in when it should not), and
``detection_rate`` is barge-in recall. Real recordings carry no frame-precise user
onset, so true stop-latency (FDB ``tstop``) is *not* computed here — that needs
controlled synthesis; ``fire_ms`` is exposed only as a clip-relative diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

CONDITION_ECHO_ONLY = "echo_only"
CONDITION_DOUBLE_TALK = "double_talk"


@dataclass(frozen=True)
class BargeInOutcome:
    """One replayed pair: what the decider did vs what it should have done."""

    condition: str
    expected_interrupt: bool
    interrupted: bool
    interrupt_ms: float | None = None
    with_movement: bool = False


@dataclass(frozen=True)
class BargeInReport:
    n_total: int
    n_echo_only: int
    n_double_talk: int
    false_interrupt_count: int
    detection_count: int
    false_interrupt_rate: float  # over echo_only  (≈ FDB Takeover Rate)
    detection_rate: float  # over double_talk (barge-in recall)
    missed_rate: float  # 1 - detection_rate
    median_fire_ms: float | None  # clip-relative, diagnostic only (NOT onset latency)
    by_movement: dict[str, dict[str, float]] = field(default_factory=dict)


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _core_counts(outcomes: list[BargeInOutcome]) -> tuple[int, int, int, int]:
    """(n_echo_only, n_double_talk, false_interrupts, detections)."""
    echo = [o for o in outcomes if o.condition == CONDITION_ECHO_ONLY]
    dbl = [o for o in outcomes if o.condition == CONDITION_DOUBLE_TALK]
    false_int = sum(1 for o in echo if o.interrupted)
    detected = sum(1 for o in dbl if o.interrupted)
    return len(echo), len(dbl), false_int, detected


def _movement_split(outcomes: list[BargeInOutcome]) -> dict[str, dict[str, float]]:
    """False-interrupt / detection rates split by static vs moving device.

    Movement perturbs the echo path mid-recording — the harder acoustic case — so
    keeping it separate shows whether AEC robustness degrades under it."""
    split: dict[str, dict[str, float]] = {}
    for label, moving in (("static", False), ("moving", True)):
        subset = [o for o in outcomes if o.with_movement == moving]
        if not subset:
            continue
        n_echo, n_dbl, false_int, detected = _core_counts(subset)
        split[label] = {
            "false_interrupt_rate": _rate(false_int, n_echo),
            "detection_rate": _rate(detected, n_dbl),
            "n_echo_only": n_echo,
            "n_double_talk": n_dbl,
        }
    return split


def barge_in_report(outcomes: list[BargeInOutcome]) -> BargeInReport:
    """Decompose barge-in outcomes into false-interrupt / detection / miss metrics."""
    n_echo, n_dbl, false_int, detected = _core_counts(outcomes)
    detection_rate = _rate(detected, n_dbl)

    fire_times = [
        o.interrupt_ms
        for o in outcomes
        if o.condition == CONDITION_DOUBLE_TALK and o.interrupted and o.interrupt_ms is not None
    ]

    return BargeInReport(
        n_total=len(outcomes),
        n_echo_only=n_echo,
        n_double_talk=n_dbl,
        false_interrupt_count=false_int,
        detection_count=detected,
        false_interrupt_rate=_rate(false_int, n_echo),
        detection_rate=detection_rate,
        missed_rate=1.0 - detection_rate if n_dbl else 0.0,
        median_fire_ms=float(median(fire_times)) if fire_times else None,
        by_movement=_movement_split(outcomes),
    )
