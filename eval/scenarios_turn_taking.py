"""Synthesize turn-taking scenarios from FLEURS utterances (P3.1 Pha B, Tier 2).

Turn-taking is measured with *no* assistant playing, so there is no echo and no RIR
to synthesize here (that is the barge-in tier's concern) — only the user's audio and
a **known ground-truth turn end**. Controlling the timeline is exactly what real
corpora cannot give us (Full-Duplex-Bench synthesises interruptions for the same
reason: public turn-taking corpora are scarce, and none are Vietnamese two-channel).

Two scenario shapes, each a ``near`` timeline plus the frames that define correctness:

    clean      [utterance][trailing silence]
               → the turn genuinely ends; measure how long each policy over-waits.
    mid_pause  [first half][pause][second half][trailing silence]
               → a WITHIN-turn pause; a good endpoint must hold the floor through it.
               Stopping inside the pause is a cut-in (premature end-of-turn).

The pause length is the lever: a pause longer than a fixed timeout but shorter than
the adaptive floor is where ``fixed`` cuts in and ``p_based`` (Smart-Turn holding the
floor) does not — the comparison the plan is built around.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from local import config as cfg

_SR = 16000
# Trailing silence must exceed the largest possible required-silence (adaptive ceil =
# 3000ms) so every policy is guaranteed to close on a clean end.
_TRAILING_SILENCE_MS = 3500.0


@dataclass(frozen=True)
class TurnScenario:
    """One user-audio timeline with the ground truth needed to score endpointing."""

    scenario_type: str  # "clean" | "mid_pause"
    fileid: str
    near: np.ndarray  # 16k float32 mono
    true_end_ms: float  # end of the final speech = the real turn end
    pause_start_ms: float | None = None  # mid_pause only
    pause_ms: float | None = None  # mid_pause only

    @property
    def pause_end_ms(self) -> float | None:
        if self.pause_start_ms is None or self.pause_ms is None:
            return None
        return self.pause_start_ms + self.pause_ms


def _silence(ms: float) -> np.ndarray:
    return np.zeros(int(ms / 1000 * _SR), dtype=np.float32)


def _trim_silence(audio: np.ndarray, *, frame: int = 512, floor_rms: float = 0.01) -> np.ndarray:
    """Strip leading/trailing low-energy frames so the audio ends on the last word.

    Read-speech clips carry padding silence; without trimming, ``true_end`` would sit
    inside that padding and understate over-wait. Internal pauses are kept — they are
    part of the turn and are exactly what the endpoint must ride through."""
    n = len(audio) // frame
    if n == 0:
        return audio
    voiced = [
        i
        for i in range(n)
        if float(np.sqrt(np.mean(audio[i * frame : (i + 1) * frame] ** 2))) > floor_rms
    ]
    if not voiced:
        return audio
    return audio[voiced[0] * frame : (voiced[-1] + 1) * frame]


def _ms(n_samples: int) -> float:
    return n_samples / _SR * 1000.0


def make_clean(fileid: str, utterance: np.ndarray, trailing_ms: float = _TRAILING_SILENCE_MS) -> TurnScenario:
    """[utterance][trailing silence]; the turn ends when the utterance ends."""
    utterance = np.asarray(utterance, dtype=np.float32).reshape(-1)
    near = np.concatenate([utterance, _silence(trailing_ms)])
    return TurnScenario(
        scenario_type="clean",
        fileid=fileid,
        near=near.astype(np.float32, copy=False),
        true_end_ms=_ms(len(utterance)),
    )


def make_mid_pause(
    fileid: str,
    utterance: np.ndarray,
    pause_ms: float,
    *,
    split_frac: float = 0.5,
    trailing_ms: float = _TRAILING_SILENCE_MS,
) -> TurnScenario:
    """[first half][pause][second half][trailing]; the turn ends after the 2nd half.

    The pause is a within-turn hesitation, not an end-of-turn: stopping inside it is
    a cut-in. Splitting mid-utterance may land inside a word — fine for an
    energy/acoustic endpoint, and it makes the pause a genuinely mid-sentence gap.
    """
    utterance = np.asarray(utterance, dtype=np.float32).reshape(-1)
    split = int(len(utterance) * split_frac)
    first, second = utterance[:split], utterance[split:]
    pause = _silence(pause_ms)
    near = np.concatenate([first, pause, second, _silence(trailing_ms)])
    pause_start = _ms(len(first))
    true_end = _ms(len(first) + len(pause) + len(second))
    return TurnScenario(
        scenario_type="mid_pause",
        fileid=fileid,
        near=near.astype(np.float32, copy=False),
        true_end_ms=true_end,
        pause_start_ms=pause_start,
        pause_ms=pause_ms,
    )


def _load_fleurs_utterances(n: int, seed: int) -> list[tuple[str, np.ndarray]]:
    """Deterministically sample n FLEURS utterances (same seed → same items)."""
    rows = [json.loads(line) for line in cfg.FLEURS_MANIFEST.open(encoding="utf-8") if line.strip()]
    random.Random(seed).shuffle(rows)
    picked = rows[:n]
    out: list[tuple[str, np.ndarray]] = []
    for row in picked:
        audio, sr = sf.read(cfg.FLEURS_WAV_DIR / row["filename"], dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != _SR:
            raise ValueError(f"{row['filename']}: expected {_SR}Hz, got {sr}")
        trimmed = _trim_silence(np.ascontiguousarray(audio, dtype=np.float32))
        out.append((row["filename"], trimmed))
    return out


def build_scenarios(n_utterances: int, pause_ms: float, seed: int = 42) -> list[TurnScenario]:
    """One clean + one mid_pause scenario per sampled utterance (paired, same audio)."""
    scenarios: list[TurnScenario] = []
    for fileid, utterance in _load_fleurs_utterances(n_utterances, seed):
        scenarios.append(make_clean(fileid, utterance))
        scenarios.append(make_mid_pause(fileid, utterance, pause_ms))
    return scenarios
