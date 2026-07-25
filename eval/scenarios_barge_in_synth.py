"""Synthesize VN barge-in scenarios with real-RIR echo (P3.1 Pha B, Tier 1 synth).

The real AEC-Challenge tier proves barge-in survives device echo, but its recordings
carry no frame-precise user onset, so *latency* cannot be read from them. This module
fills that gap: it places a Vietnamese barge-in at a **known onset** over echo built
from real MIT impulse responses, so ``latency = fire − onset`` is measurable, and adds
the backchannel probe the real corpus lacks.

    near(t) = alpha · (far * RIR)  [ + user(t − onset) ]

``far`` (the assistant speaking) and ``user`` (the barge-in) are both FLEURS clips —
content is irrelevant to an energy/echo front-end, so no TTS is needed here; the RIR
supplies real reverberation (MIT IR Survey, ~1.5m spacing) rather than a toy delay.

Three kinds:
    echo_only    far echo, no user           → MUST NOT interrupt (false-interrupt)
    barge_in     far echo + user @ onset      → SHOULD interrupt  (detection + latency)
    backchannel  far echo + ~400ms "uh" @ onset → ideally NOT (a short acknowledgement).
                 Expected finding: the sustained-energy gate fires anyway — motivating a
                 future backchannel classifier. Honest, not hidden.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from local import config as cfg

_SR = 16000
_FRAME = 512
_RIR_MANIFEST = Path("data/rir/mit/manifest.jsonl")
_RIR_DIR = Path("data/rir/mit")


@dataclass(frozen=True)
class SynthScenario:
    """A synthesized (far, near) pair with a known onset for latency scoring."""

    kind: str  # echo_only | barge_in | backchannel
    far: np.ndarray  # speaker reference (16k float32)
    near: np.ndarray  # microphone = echo [+ user]; same length as far
    expected_interrupt: bool
    onset_ms: float | None  # user onset (barge_in / backchannel); None for echo_only


def synth_echo(far: np.ndarray, rir: np.ndarray, alpha: float) -> np.ndarray:
    """Convolve far with a real RIR, truncate to far's length, scale by alpha."""
    echo = fftconvolve(far, rir)[: len(far)]
    return (echo * alpha).astype(np.float32)


def _normalize_rms(audio: np.ndarray, target_rms: float) -> np.ndarray:
    """Scale to a target RMS so scenario levels (and thus SER) are controlled, not
    left to each raw FLEURS clip — some of which are far below speech level."""
    current = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
    if current < 1e-8:
        return audio.astype(np.float32)
    scaled = audio * (target_rms / current)
    peak = float(np.max(np.abs(scaled)))
    if peak > 0.99:
        scaled = scaled * (0.99 / peak)
    return scaled.astype(np.float32)


def _trim_silence(audio: np.ndarray, *, floor_rms: float = 0.01) -> np.ndarray:
    n = len(audio) // _FRAME
    if n == 0:
        return audio
    voiced = [
        i
        for i in range(n)
        if float(np.sqrt(np.mean(audio[i * _FRAME : (i + 1) * _FRAME] ** 2))) > floor_rms
    ]
    if not voiced:
        return audio
    return audio[voiced[0] * _FRAME : (voiced[-1] + 1) * _FRAME]


def _align(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    n = min(len(a) for a in arrays)
    n -= n % _FRAME
    return tuple(np.ascontiguousarray(a[:n], dtype=np.float32) for a in arrays)


def _place(length: int, segment: np.ndarray, onset_sample: int) -> np.ndarray:
    track = np.zeros(length, dtype=np.float32)
    end = min(length, onset_sample + len(segment))
    if end > onset_sample:
        track[onset_sample:end] = segment[: end - onset_sample]
    return track


def _pad_to(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) >= length:
        return audio[:length]
    return np.concatenate([audio, np.zeros(length - len(audio), dtype=np.float32)])


def make_echo_only(far: np.ndarray, rir: np.ndarray, alpha: float) -> SynthScenario:
    near = synth_echo(far, rir, alpha)
    far_a, near_a = _align(far, near)
    return SynthScenario("echo_only", far_a, near_a, expected_interrupt=False, onset_ms=None)


def _echo_plus_segment(
    far: np.ndarray, rir: np.ndarray, segment: np.ndarray, onset_ms: float, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    onset_sample = int(onset_ms / 1000 * _SR)
    length = max(len(far), onset_sample + len(segment))
    far_p = _pad_to(far, length)
    near = synth_echo(far_p, rir, alpha) + _place(length, segment, onset_sample)
    return _align(far_p, near)


def make_barge_in(
    far: np.ndarray, rir: np.ndarray, user: np.ndarray, onset_ms: float, alpha: float
) -> SynthScenario:
    far_a, near_a = _echo_plus_segment(far, rir, user, onset_ms, alpha)
    return SynthScenario("barge_in", far_a, near_a, expected_interrupt=True, onset_ms=onset_ms)


def make_backchannel(
    far: np.ndarray, rir: np.ndarray, backchannel: np.ndarray, onset_ms: float, alpha: float
) -> SynthScenario:
    far_a, near_a = _echo_plus_segment(far, rir, backchannel, onset_ms, alpha)
    return SynthScenario(
        "backchannel", far_a, near_a, expected_interrupt=False, onset_ms=onset_ms
    )


def _load_wav(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != _SR:
        raise ValueError(f"{path.name}: expected {_SR}Hz, got {sr}")
    return np.ascontiguousarray(audio, dtype=np.float32)


def _load_fleurs(n: int, seed: int) -> list[np.ndarray]:
    rows = [json.loads(line) for line in cfg.FLEURS_MANIFEST.open(encoding="utf-8") if line.strip()]
    random.Random(seed).shuffle(rows)
    return [_trim_silence(_load_wav(cfg.FLEURS_WAV_DIR / r["filename"])) for r in rows[:n]]


def _load_rirs(seed: int) -> list[np.ndarray]:
    rows = [json.loads(line) for line in _RIR_MANIFEST.open(encoding="utf-8") if line.strip()]
    random.Random(seed).shuffle(rows)
    return [_load_wav(_RIR_DIR / r["filename"]) for r in rows]


def build_scenarios(
    n: int,
    *,
    onset_ms: float = 1000.0,
    alpha: float = 0.5,
    backchannel_ms: float = 400.0,
    speech_rms: float = 0.1,
    seed: int = 42,
) -> list[SynthScenario]:
    """Per index: one echo_only + one barge_in + one backchannel (paired far/RIR).

    ``far`` = assistant clip, ``user`` = a different clip (the barge-in), ``backchannel``
    = the user clip's first ``backchannel_ms`` (an "uh/vâng"-length acknowledgement).
    Both far and user are RMS-normalised to ``speech_rms`` so the signal-to-echo ratio
    is set by ``alpha`` (not by whichever raw FLEURS clip was drawn — some sit far below
    speech level). RIRs cycle deterministically so every run is reproducible."""
    if not _RIR_MANIFEST.exists():
        raise FileNotFoundError(
            f"RIR manifest missing: {_RIR_MANIFEST}. Run "
            "uv run --with pyarrow python scripts/download_rir.py"
        )
    fars = _load_fleurs(n, seed)
    users = _load_fleurs(n, seed + 1)  # disjoint sampling for the barge-in voice
    rirs = _load_rirs(seed)
    bc_len = int(backchannel_ms / 1000 * _SR)

    scenarios: list[SynthScenario] = []
    for i in range(min(len(fars), len(users))):
        far = _normalize_rms(fars[i], speech_rms)
        user = _normalize_rms(users[i], speech_rms)
        rir = rirs[i % len(rirs)]
        backchannel = user[:bc_len]
        scenarios.append(make_echo_only(far, rir, alpha))
        scenarios.append(make_barge_in(far, rir, user, onset_ms, alpha))
        scenarios.append(make_backchannel(far, rir, backchannel, onset_ms, alpha))
    return scenarios
