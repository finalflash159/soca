"""Stage-decomposition metrics for the RobustASR ablation (P1.1 Pha A).

The Table VII harness (``local/eval_table7.py``) already records per-item
diagnostics but only summarises WER / CER / hallucination-rate / latency. That
summary cannot answer the question the paper's Table VII is about: *which mitigation
stage caught what, and at what false-reject cost*.

This module turns a list of diagnostics into that decomposition — pure functions
over plain data, so it is testable without a model or audio.

Canonical stages (pipeline order) mirror ``RobustASR.transcribe``:
    no_speech        — Silero VAD gate (stage 1)
    empty_asr        — model produced nothing (stage 2)
    low_confidence   — avg-logprob guard (stage 2b)
    high_compression — compression-ratio guard (stage 2b)
    empty_after_boh  — BoH cleaning emptied the text (stage 4)
    heuristic        — heuristic safety net (stage 5)
    empty_output     — empty with no reason (a silent catch, manual configs)
    accepted         — survived every stage (a hallucination when the input was noise)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from jiwer import cer, wer

STAGE_ORDER: tuple[str, ...] = (
    "no_speech",
    "empty_asr",
    "low_confidence",
    "high_compression",
    "empty_after_boh",
    "heuristic",
    "empty_output",
    "accepted",
)


@dataclass(frozen=True)
class Diagnostic:
    """One item's outcome, adapted from the harness' per-item diagnostics."""

    kind: str  # "speech" | "noise"
    rejection_reason: str
    final_text: str
    ground_truth: str = ""
    subtype: str = "unknown"


@dataclass(frozen=True)
class RobustnessReport:
    n_speech: int
    n_noise: int
    false_reject_count: int
    false_reject_rate: float
    hallucination_count: int
    hallucination_rate: float
    catch_rate: float
    wer: float | None
    cer: float | None
    noise_stage_breakdown: dict[str, int] = field(default_factory=dict)
    hallucination_rate_by_subtype: dict[str, float] = field(default_factory=dict)


def canonical_stage(rejection_reason: str, final_text: str) -> str:
    """Map a (possibly suffixed) rejection reason to a canonical stage bucket.

    ``low_confidence:-1.23`` → ``low_confidence``. An empty reason means the item
    was not rejected: ``accepted`` if text survived, else ``empty_output``.
    """
    reason = (rejection_reason or "").strip()
    if reason:
        return reason.split(":", 1)[0]
    return "accepted" if (final_text or "").strip() else "empty_output"


def _is_hallucination(diag: Diagnostic) -> bool:
    return diag.kind == "noise" and bool((diag.final_text or "").strip())


def _is_false_reject(diag: Diagnostic) -> bool:
    return diag.kind == "speech" and not (diag.final_text or "").strip()


def noise_stage_breakdown(diagnostics: list[Diagnostic]) -> dict[str, int]:
    """Count noise items by the stage that caught them, in pipeline order.

    The ``accepted`` bucket is the leak-through count (hallucinations)."""
    counts = Counter(
        canonical_stage(d.rejection_reason, d.final_text)
        for d in diagnostics
        if d.kind == "noise"
    )
    # Known stages first in pipeline order, then any unrecognised reason appended
    # rather than silently dropped, so the breakdown always sums to n_noise.
    ordered = {stage: counts[stage] for stage in STAGE_ORDER if counts[stage]}
    extras = {s: c for s, c in counts.items() if s not in STAGE_ORDER and c}
    return {**ordered, **extras}


def _accepted_speech_wer_cer(
    diagnostics: list[Diagnostic],
) -> tuple[float | None, float | None]:
    refs: list[str] = []
    preds: list[str] = []
    for d in diagnostics:
        if d.kind != "speech":
            continue
        pred = (d.final_text or "").strip()
        if not pred:  # false-reject; accounted separately, not as recognition error
            continue
        refs.append((d.ground_truth or "").lower().strip())
        preds.append(pred.lower())
    if not refs:
        return None, None
    return float(wer(refs, preds)), float(cer(refs, preds))


def _hallucination_rate_by_subtype(diagnostics: list[Diagnostic]) -> dict[str, float]:
    totals: Counter[str] = Counter()
    leaks: Counter[str] = Counter()
    for d in diagnostics:
        if d.kind != "noise":
            continue
        totals[d.subtype] += 1
        if _is_hallucination(d):
            leaks[d.subtype] += 1
    return {sub: leaks[sub] / totals[sub] for sub in totals}


def compute_robustness_report(diagnostics: list[Diagnostic]) -> RobustnessReport:
    """Decompose a config's diagnostics into stage / false-reject / catch metrics."""
    speech = [d for d in diagnostics if d.kind == "speech"]
    noise = [d for d in diagnostics if d.kind == "noise"]

    false_rejects = sum(1 for d in speech if _is_false_reject(d))
    hallucinations = sum(1 for d in noise if _is_hallucination(d))

    false_reject_rate = false_rejects / len(speech) if speech else 0.0
    hallucination_rate = hallucinations / len(noise) if noise else 0.0

    wer_val, cer_val = _accepted_speech_wer_cer(speech)

    return RobustnessReport(
        n_speech=len(speech),
        n_noise=len(noise),
        false_reject_count=false_rejects,
        false_reject_rate=false_reject_rate,
        hallucination_count=hallucinations,
        hallucination_rate=hallucination_rate,
        catch_rate=1.0 - hallucination_rate if noise else 0.0,
        wer=wer_val,
        cer=cer_val,
        noise_stage_breakdown=noise_stage_breakdown(noise),
        hallucination_rate_by_subtype=_hallucination_rate_by_subtype(noise),
    )
