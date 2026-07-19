from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Silero streaming frame @16kHz: exactly 512 samples = 32ms (same as DuplexAecSink).
VAD_FRAME_SAMPLES = 512
VAD_FRAME_MS = 32.0
# A transient (cough, keyboard click) rarely sustains 3 consecutive frames (~96ms);
# require this streak before treating audio as speech onset.
SPEECH_CONFIRM_FRAMES = 3

# --- partial-caption cadence tuning (duty-cycle) ---
PARTIAL_MARGIN = 2.0        # interval >= margin * per_call  → duty-cycle <= 1/margin
PARTIAL_FLOOR_MS = 800      # below this, faster than a human reads a phrase
PARTIAL_CEIL_MS = 2000      # above this, captions stop being useful
PARTIAL_DISABLE_MS = 1200   # effective per_call above this -> partials useless -> disable
PARTIAL_EWMA_UP = 0.5       # slower: widen interval FAST (do not starve CPU)
PARTIAL_EWMA_DOWN = 0.2     # faster again: tighten interval SLOWLY

def clamp_interval_ms(value_ms: float) -> int:
    return int(round(max(PARTIAL_FLOOR_MS, min(PARTIAL_CEIL_MS, value_ms))))

def partial_interval_from_cost(
    per_call_ms: float, cpu_count: int | None = None
) -> tuple[int, bool]:
    """(interval_ms, enabled) from IDLE-measured transcribe cost (Tier 1 seed).

    Multiply by rho (oversubscription from num_threads=4 vs cores) BECAUSE the idle
    measurement excludes contention. Tier 2 (online) measures real wall-time -> NO rho again.
    """
    cores = max(1, cpu_count or 1)
    rho = max(1.0, 4.0 / cores)
    effective = per_call_ms * rho
    if effective > PARTIAL_DISABLE_MS:
        return PARTIAL_CEIL_MS, False
    return clamp_interval_ms(PARTIAL_MARGIN * effective), True


class IncrementalVadTracker:
    """O(n) streaming VAD state: each mic frame passes through Silero ONCE.

    Replaces the O(n²) "re-scan the whole buffer every block" pattern. Exposes
    the two numbers the endpoint decision needs, updated per frame:
      - speech_ms      : how much confirmed speech we heard this turn
      - silence_run_ms : how long it has been silent SINCE the last speech
    """

    def __init__(self, model, *, threshold: float = 0.5) -> None:
        self._model = model            # Silero VAD (stateful), injected
        self.threshold = threshold
        self._residual = np.empty(0, dtype=np.float32)  # <512-sample leftover
        self._streak = 0               # consecutive speech-frames counter
        self.speech_ms = 0.0
        self.silence_run_ms = 0.0
        self.has_speech = False

    def reset(self) -> None:
        """Call at the start of every turn — Silero keeps internal RNN state."""
        if hasattr(self._model, "reset_states"):
            self._model.reset_states()
        self._residual = np.empty(0, dtype=np.float32)
        self._streak = 0
        self.speech_ms = 0.0
        self.silence_run_ms = 0.0
        self.has_speech = False

    def seed_speech(self, ms: float) -> None:
        """Barge-in carry-over: the prefix already contains real speech."""
        self.speech_ms += max(0.0, ms)
        self.has_speech = True
        self.silence_run_ms = 0.0

    def feed(self, block: np.ndarray) -> None:
        """Consume one mic block (any length); process complete 512-frames."""
        import torch

        samples = np.concatenate(
            [self._residual, np.asarray(block, dtype=np.float32).reshape(-1)]
        )
        n_frames = len(samples) // VAD_FRAME_SAMPLES
        for i in range(n_frames):
            frame = samples[i * VAD_FRAME_SAMPLES : (i + 1) * VAD_FRAME_SAMPLES]
            prob = float(self._model(torch.from_numpy(frame), 16000).item())
            if prob >= self.threshold:
                self._streak += 1
                if self._streak == SPEECH_CONFIRM_FRAMES:
                    # Onset confirmed: retroactively count the confirm window.
                    self.has_speech = True
                    self.speech_ms += VAD_FRAME_MS * SPEECH_CONFIRM_FRAMES
                    self.silence_run_ms = 0.0
                elif self._streak > SPEECH_CONFIRM_FRAMES:
                    self.speech_ms += VAD_FRAME_MS
                    self.silence_run_ms = 0.0
            else:
                self._streak = 0
                if self.has_speech:
                    self.silence_run_ms += VAD_FRAME_MS
        self._residual = samples[n_frames * VAD_FRAME_SAMPLES :]


def required_silence_ms(
    speech_ms: float,
    config,
    *,
    incomplete: bool = False,
) -> float:
    """How much trailing silence closes the turn, given how much was spoken.

    adaptive=False -> the legacy fixed threshold (regression-safe).
    Short utterance  -> patient (the user is warming up / thinking).
    Long utterance   -> eager  (they said plenty; clear silence means done).
    Between          -> linear interpolation patient -> eager.
    incomplete=True  -> add a hold (partial text ends mid-thought).
    """
    if not getattr(config, "adaptive", False):
        return float(config.endpoint_silence_ms)

    short = float(config.short_speech_ms)
    long_ = float(config.long_speech_ms)
    patient = float(config.patient_silence_ms)
    eager = float(config.eager_silence_ms)

    if speech_ms <= short:
        base = patient
    elif speech_ms >= long_:
        base = eager
    else:
        t = (speech_ms - short) / (long_ - short)
        base = patient + (eager - patient) * t

    if incomplete:
        base += float(config.incomplete_hold_ms)
    return base


# Conservative list: only words that almost never END a complete Vietnamese
# sentence. Deliberately EXCLUDES "rồi" ("mấy giờ rồi") and "hay" ("cái này hay").
_INCOMPLETE_WORDS = frozenset(
    "và nhưng mà thì với hoặc của cho để vì nên còn là bị được như kiểu "
    "ừ à ờ ừm ơ".split()
)
_INCOMPLETE_PAIRS = frozenset({"tức là", "nghĩa là", "như là", "tại vì", "bởi vì"})
_SENTENCE_END = ".!?…"


def is_incomplete_vietnamese(text: str) -> bool:
    """Heuristic: does this partial transcript look mid-thought?

    Used ONLY to EXTEND patience (hold), never as the sole early-cut signal —
    ASR on audio cut mid-word can garble the tail.
    """
    cleaned = text.strip().lower()
    if not cleaned:
        return False
    if cleaned[-1] == ",":
        return True
    if cleaned[-1] in _SENTENCE_END:
        return False
    words = cleaned.replace(",", " ").split()
    if not words:
        return False
    if len(words) >= 2 and f"{words[-2]} {words[-1]}" in _INCOMPLETE_PAIRS:
        return True
    return words[-1] in _INCOMPLETE_WORDS


@dataclass
class LocalAgreement:
    """LocalAgreement-2 (ufal/whisper_streaming): commit the longest common
    word-prefix of two consecutive decodes; only the tail may still change.

    committed is monotonic (never shrinks) so the caption a user already read
    never jumps back. The FINAL transcript is RobustASR's job, not ours.
    """

    _prev: list[str] = field(default_factory=list)
    committed: list[str] = field(default_factory=list)

    def update(self, text: str) -> tuple[str, str]:
        words = text.split()
        common: list[str] = []
        for old, new in zip(self._prev, words, strict=False):
            if old != new:
                break
            common.append(new)
        if len(common) > len(self.committed):
            self.committed = common
        self._prev = words
        tentative = words[len(self.committed) :]
        return " ".join(self.committed), " ".join(tentative)

    def reset(self) -> None:
        self._prev = []
        self.committed = []


# --- P-based endpointing: dynamic silence threshold from P(still speaking) ---
# The constants below are CALIBRATION TARGETS (tuned by local/eval_endpoint.py),
# not magic numbers. Design + sources: zplan/endpointing_research.vi.md.
P_INCOMPLETE_WEIGHT = 0.8            # text ends on a connective -> strong "still going"
_ENERGY_EDGE_MS = 120               # trailing window for abrupt-vs-tapered energy
_ENERGY_R_LO, _ENERGY_R_HI = 0.3, 0.9      # edge/ref ratio -> score
_FILLER_FLUX_LO, _FILLER_FLUX_HI = 0.02, 0.06   # spectral flux: <=LO => held vowel
_FILLER_DUR_LO, _FILLER_DUR_HI = 250.0, 450.0   # ms of steady sound for a filler
_STFT_FRAME, _STFT_HOP = 320, 160   # 20ms window / 10ms hop @16kHz


def _rms_frames(x: np.ndarray, frame: int = _STFT_FRAME) -> np.ndarray:
    n = len(x) // frame
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    framed = x[: n * frame].reshape(n, frame)
    return np.sqrt((framed * framed).mean(axis=1) + 1e-12)


def energy_trailing_score(
    voiced_tail: np.ndarray, *, edge_ms: int = _ENERGY_EDGE_MS, sr: int = 16000
) -> float:
    """Acoustic cue via amplitude. 1.0 = voice stopped at ~full loudness (abrupt ->
    maybe still going); 0.0 = voice tapered to near-silence (final lowering -> done).

    voiced_tail is the speech right before the current pause. Proxy for prosodic
    final-lowering; weak on its own (loud emphatic endings false-positive).
    """
    x = np.asarray(voiced_tail, dtype=np.float32).reshape(-1)
    rms = _rms_frames(x)
    if len(rms) < 4:
        return 0.0
    ref = float(np.median(rms))                 # typical voiced loudness (robust)
    k = max(1, int(edge_ms * sr / 1000) // _STFT_FRAME)
    edge = float(rms[-k:].mean())               # loudness right before the pause
    ratio = edge / (ref + 1e-9)
    return float(
        np.clip((ratio - _ENERGY_R_LO) / (_ENERGY_R_HI - _ENERGY_R_LO), 0.0, 1.0)
    )


def _stft_mag(
    x: np.ndarray, frame: int = _STFT_FRAME, hop: int = _STFT_HOP
) -> np.ndarray | None:
    n = 1 + (len(x) - frame) // hop
    if n < 2:
        return None
    window = np.hanning(frame).astype(np.float32)
    cols = np.empty((n, frame // 2 + 1), dtype=np.float32)
    for i in range(n):
        cols[i] = np.abs(np.fft.rfft(x[i * hop : i * hop + frame] * window))
    return cols


def acoustic_filler_score(voiced_tail: np.ndarray, *, sr: int = 16000) -> float:
    """Acoustic cue via spectral steadiness. 1.0 = tail is a LONG, spectrally STEADY
    sound (hesitation 'ummm': the vocal tract is held, so the spectrum barely moves);
    0.0 = normal articulated speech (formants move -> high spectral flux).

    Independent of ASR text, so it catches held hesitations the transcript misses.
    Requires steady spectrum AND long AND sustained energy: a *fading* final vowel
    also has a steady spectrum, so the sustain gate keeps it from looking like a hum.
    """
    x = np.asarray(voiced_tail, dtype=np.float32).reshape(-1)
    mag = _stft_mag(x)
    if mag is None:
        return 0.0
    norm = mag / (mag.sum(axis=1, keepdims=True) + 1e-9)     # per-frame shape
    flux = np.sqrt((np.diff(norm, axis=0) ** 2).sum(axis=1))  # frame-to-frame change
    flux_mean = float(flux.mean())                           # LOW => held vowel
    dur_ms = len(x) / sr * 1000.0
    steady = np.clip(
        (_FILLER_FLUX_HI - flux_mean) / (_FILLER_FLUX_HI - _FILLER_FLUX_LO), 0.0, 1.0
    )
    long_ = np.clip(
        (dur_ms - _FILLER_DUR_LO) / (_FILLER_DUR_HI - _FILLER_DUR_LO), 0.0, 1.0
    )
    sustain = energy_trailing_score(x, sr=sr)                # 0 if fading out
    return float(steady * long_ * sustain)                  # steady AND long AND held


def estimate_p_still_speaking(
    partial_text: str, voiced_tail: np.ndarray | None, *, sr: int = 16000
) -> float:
    """Fuse cheap cues into P(user is still speaking) in [0, 1].

    Combine with max (not sum/noisy-OR): the text-connective cue and text-filler cue
    are correlated, so max avoids double-counting. Baseline 0 = default eager; evidence
    only raises patience. This is a hand-built stand-in for a trained turn model
    (Smart Turn / LiveKit) — same slot, swappable later.
    """
    p = 0.0
    if is_incomplete_vietnamese(partial_text or ""):
        p = max(p, P_INCOMPLETE_WEIGHT)
    if voiced_tail is not None and len(voiced_tail) > 0:
        p = max(p, energy_trailing_score(voiced_tail, sr=sr))
        p = max(p, acoustic_filler_score(voiced_tail, sr=sr))
    return float(min(1.0, p))


def required_silence_from_p(p_still: float, config) -> float:
    """Dynamic end-of-turn silence: floor + span * P(still speaking).

    The production pattern (Vapi default waitFunction = 200 + 8000*x, LiveKit
    turn-detector modulating VAD timeout). P near 0 -> respond fast; near 1 -> patient.
    """
    floor = float(config.floor_silence_ms)
    ceil = float(config.ceil_silence_ms)
    p = max(0.0, min(1.0, float(p_still)))
    return floor + (ceil - floor) * p
