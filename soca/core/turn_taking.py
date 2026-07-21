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

def required_silence_from_p(p_still: float, config) -> float:
    """Dynamic end-of-turn silence: floor + span * P(still speaking).

    The production pattern (Vapi default waitFunction = 200 + 8000*x, LiveKit
    turn-detector modulating VAD timeout). P near 0 -> respond fast; near 1 -> patient.
    """
    floor = float(config.floor_silence_ms)
    ceil = float(config.ceil_silence_ms)
    p = max(0.0, min(1.0, float(p_still)))
    return floor + (ceil - floor) * p
