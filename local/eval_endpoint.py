"""Eval endpoint policies: fixed vs Smart Turn, on REAL speech.

Why this design (see zplan/endpointing_research.vi.md §5): the previous eval fed
whole multi-sentence FLEURS paragraphs, so the endpointer stopped at the first
sentence boundary — correct turn-taking behaviour, but mislabelled "early cut".
This version instead measures two clean, separable things on controlled clips:

  1. false_cut_rate(G): take ONE clean phrase, split it at a mid-phrase energy dip
     (an inter-word boundary, so no word is chopped), insert G ms of silence, then
     the rest. A "false cut" = the endpoint closes during the gap, before the second
     half arrives. Swept over G to draw a gradient, not a single point.

  2. end_latency: on a clean phrase with NO inserted gap, how long after real speech
     ends does the turn close? (responsiveness)

Modes compared: fixed (700ms) · smart_turn (floor + span*P(still speaking)).
Runs headless, no mic. Smart Turn is loaded once and injected into the endpoint.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import click
import numpy as np
import soundfile as sf

from soca.asr.vad import SpeechDetector
from soca.core.endpoint import EndpointConfig, record_until_silence
from soca.core.smart_turn import SmartTurnDetector

SR = 16000
TAIL_S = 3.5          # long enough to observe a close even at the P ceiling (3000ms)
_FRAME = 320          # 20ms, for the energy-dip split


class ArrayStream:
    """Fake InputStream: emits audio block-by-block, pads silence when exhausted."""

    def __init__(self, audio: np.ndarray, block: int):
        self._audio = audio
        self._pos = 0
        self._block = block

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n: int):
        chunk = self._audio[self._pos : self._pos + n]
        self._pos += n
        if len(chunk) < n:
            chunk = np.pad(chunk, (0, n - len(chunk)))
        return chunk.reshape(-1, 1), False


def _rms_frames(x: np.ndarray) -> np.ndarray:
    n = len(x) // _FRAME
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    f = x[: n * _FRAME].reshape(n, _FRAME)
    return np.sqrt((f * f).mean(axis=1) + 1e-12)


def clean_phrase(audio: np.ndarray, detector: SpeechDetector) -> np.ndarray | None:
    """First contiguous VAD speech segment = one whole phrase (starts/ends on a
    natural boundary, so no word is chopped)."""
    ts = detector.speech_timestamps(audio)
    if not ts:
        return None
    seg = audio[ts[0]["start"] : ts[0]["end"]]
    return seg if len(seg) >= SR // 2 else None      # need >= 0.5s to be useful


def split_at_dip(phrase: np.ndarray, lo=0.45, hi=0.70) -> int:
    """Split index at the quietest frame in the middle region = an inter-word dip,
    so the first half ends between words (mid-thought) without cutting a word."""
    rms = _rms_frames(phrase)
    a, b = int(lo * len(rms)), int(hi * len(rms))
    if b <= a + 1:
        return len(phrase) // 2
    return (a + int(np.argmin(rms[a:b]))) * _FRAME


def _silence(ms: float, noise: float) -> np.ndarray:
    n = int(SR * ms / 1000)
    if noise <= 0:
        return np.zeros(n, dtype=np.float32)
    return (np.random.randn(n) * noise).astype(np.float32)


def _config(mode: str) -> EndpointConfig:
    if mode == "fixed":
        return EndpointConfig(adaptive=False, max_record_ms=30000)
    if mode == "smart_turn":
        return EndpointConfig(adaptive=True, max_record_ms=30000)
    raise ValueError(f"unknown endpoint mode: {mode}")


_DETECTOR: SpeechDetector | None = None
_TURN_DETECTOR: SmartTurnDetector | None = None
_SMART_TURN_LATENCIES_MS: list[float] = []


class TimedTurnDetector:
    def __init__(self, detector: SmartTurnDetector) -> None:
        self.detector = detector

    def p_still_speaking(self, audio_window: np.ndarray) -> float:
        t0 = time.perf_counter()
        try:
            return self.detector.p_still_speaking(audio_window)
        finally:
            _SMART_TURN_LATENCIES_MS.append((time.perf_counter() - t0) * 1000)


def _record(audio: np.ndarray, mode: str) -> np.ndarray:
    kwargs = {}
    if mode == "smart_turn":
        kwargs["turn_detector"] = _TURN_DETECTOR
    return record_until_silence(
        _DETECTOR,
        config=_config(mode),
        stream_factory=lambda **k: ArrayStream(audio, 1600),
        **kwargs,
    )


def run_false_cut(phrase, gap_ms, mode, noise) -> bool:
    """True = closed during the gap (before the 2nd half) = a false cut."""
    split = split_at_dip(phrase)
    part_a, part_b = phrase[:split], phrase[split:]
    audio = np.concatenate([
        part_a, _silence(gap_ms, noise), part_b, _silence(TAIL_S * 1000, noise)
    ])
    end_of_speech = len(part_a) + int(SR * gap_ms / 1000) + len(part_b)
    got = _record(audio, mode)
    return len(got) < end_of_speech


def run_end_latency(phrase, mode, noise) -> float | None:
    """ms from real speech end to close, on a clean phrase (no inserted gap)."""
    audio = np.concatenate([phrase, _silence(TAIL_S * 1000, noise)])
    got = _record(audio, mode)
    if len(got) < len(phrase):
        return None                                  # cut inside the phrase -> skip
    return (len(got) - len(phrase)) / (SR / 1000)


@click.command()
@click.option("--n", default=24, help="number of FLEURS phrases")
@click.option("--gaps", default="300,500,700,900,1200", help="hesitation gaps (ms)")
@click.option("--noise", default=0.0, help="silence noise floor amplitude (0 = pure)")
def main(n: int, gaps: str, noise: float) -> None:
    global _DETECTOR, _TURN_DETECTOR, _SMART_TURN_LATENCIES_MS
    from local import config as cfg

    _DETECTOR = SpeechDetector()
    detector = SmartTurnDetector(
        Path(__file__).resolve().parents[1] / "models" / "smart-turn-v3-onnx"
    )
    detector.warmup()
    _TURN_DETECTOR = TimedTurnDetector(detector)
    _SMART_TURN_LATENCIES_MS = []
    gap_list = [int(g) for g in gaps.split(",")]
    modes = ("fixed", "smart_turn")

    wavs = sorted(Path(cfg.FLEURS_WAV_DIR).glob("*.wav"))
    phrases: list[np.ndarray] = []
    for p in wavs:
        if len(phrases) >= n:
            break
        a, _ = sf.read(str(p), dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        ph = clean_phrase(a.astype(np.float32), _DETECTOR)
        if ph is not None:
            phrases.append(ph)
    if not phrases:
        raise SystemExit(f"No usable phrases in {cfg.FLEURS_WAV_DIR}")

    print(f"phrases={len(phrases)}  smart_turn=on  noise={noise}\n")

    # --- Metric 1: false-cut rate vs gap ---
    print("false_cut_rate (lower = more patient, does NOT chop hesitation)")
    header = "gap_ms  | " + " | ".join(f"{m:>8}" for m in modes)
    print(header + "\n" + "-" * len(header))
    fc: dict[str, dict[int, float]] = {m: {} for m in modes}
    for g in gap_list:
        cells = []
        for m in modes:
            rate = sum(run_false_cut(ph, g, m, noise) for ph in phrases) / len(phrases)
            fc[m][g] = round(rate, 3)
            cells.append(f"{rate:7.0%} ")
        print(f"{g:6d}  | " + " | ".join(cells))

    # --- Metric 2: end-of-turn latency on clean phrases ---
    print("\nend_latency_ms (lower = snappier when the user IS done)")
    lat: dict[str, float | None] = {}
    for m in modes:
        vals = [run_end_latency(ph, m, noise) for ph in phrases]
        vals = [v for v in vals if v is not None]
        lat[m] = round(sum(vals) / len(vals), 0) if vals else None
        print(f"  {m:>8}: {lat[m]}ms  (n={len(vals)})")
    smart_turn_cpu_ms = (
        round(sum(_SMART_TURN_LATENCIES_MS) / len(_SMART_TURN_LATENCIES_MS), 2)
        if _SMART_TURN_LATENCIES_MS
        else None
    )
    print(f"\nsmart_turn_cpu_ms: {smart_turn_cpu_ms}  (n={len(_SMART_TURN_LATENCIES_MS)})")

    out = Path(cfg.EVAL_RESULTS_DIR) / "endpoint_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "phrases": len(phrases),
                "smart_turn": True,
                "noise": noise,
                "false_cut_rate": fc,
                "end_latency_ms": lat,
                "smart_turn_cpu_ms": smart_turn_cpu_ms,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
