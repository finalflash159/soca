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


# --------------------------------------------------------------------------- #
# Tier 2 — turn-taking (endpoint policy: over-wait on clean ends, cut-in on pauses)
# --------------------------------------------------------------------------- #

SCENARIO_CLEAN = "clean"
SCENARIO_MID_PAUSE = "mid_pause"


@dataclass(frozen=True)
class TurnOutcome:
    """One endpoint replay: when a policy closed the turn vs the ground truth."""

    scenario_type: str  # clean | mid_pause
    policy: str
    stopped: bool
    stop_ms: float | None
    true_end_ms: float
    pause_end_ms: float | None = None  # mid_pause only


def _over_wait_ms(outcome: TurnOutcome) -> float | None:
    """How long after the real turn end the policy kept waiting (patience cost)."""
    if outcome.stop_ms is None:
        return None
    return outcome.stop_ms - outcome.true_end_ms


def _is_cut_in(outcome: TurnOutcome) -> bool:
    """Closed *inside* a within-turn pause → premature end-of-turn (a cut-in error)."""
    if outcome.scenario_type != SCENARIO_MID_PAUSE or not outcome.stopped:
        return False
    if outcome.stop_ms is None or outcome.pause_end_ms is None:
        return False
    return outcome.stop_ms < outcome.pause_end_ms


def _is_premature_clean(outcome: TurnOutcome) -> bool:
    """Clean turn closed *before* the user finished → the endpoint fired too early
    (typically into a natural intra-sentence pause of read speech)."""
    if outcome.scenario_type != SCENARIO_CLEAN or not outcome.stopped:
        return False
    return outcome.stop_ms is not None and outcome.stop_ms < outcome.true_end_ms


def turn_taking_report(outcomes: list[TurnOutcome]) -> dict[str, dict[str, float | int | None]]:
    """Per-policy accuracy/patience split, keeping early and late errors distinct.

    - ``cut_in_rate``      : mid-pause turns closed inside the within-turn pause.
    - ``premature_close_rate``: clean turns closed before the real end (early error).
    - ``median_over_wait_ms``: over-wait on *correct* clean closes only (stop ≥ end),
      so an early stop never masquerades as a (negative) over-wait.
    Together these are the trade-off axis: eager policies fire early (cut-in /
    premature); patient ones wait past the end."""
    policies = sorted({o.policy for o in outcomes})
    report: dict[str, dict[str, float | int | None]] = {}
    for policy in policies:
        rows = [o for o in outcomes if o.policy == policy]
        clean = [o for o in rows if o.scenario_type == SCENARIO_CLEAN]
        mids = [o for o in rows if o.scenario_type == SCENARIO_MID_PAUSE]
        cut_ins = sum(1 for o in mids if _is_cut_in(o))
        prematures = sum(1 for o in clean if _is_premature_clean(o))
        # Over-wait only where the policy actually waited past the true end.
        over_waits = [
            w
            for o in clean
            if o.stopped and not _is_premature_clean(o) and (w := _over_wait_ms(o)) is not None
        ]
        report[policy] = {
            "n_clean": len(clean),
            "n_mid_pause": len(mids),
            "cut_in_rate": _rate(cut_ins, len(mids)),
            "cut_in_count": cut_ins,
            "premature_close_rate": _rate(prematures, len(clean)),
            "premature_close_count": prematures,
            "median_over_wait_ms": float(median(over_waits)) if over_waits else None,
            "n_correct_close": len(over_waits),
        }
    return report
