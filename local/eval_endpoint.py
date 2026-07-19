"""Eval endpoint policies: fixed vs length-adaptive vs P-based, on REAL speech.

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

Modes compared: fixed (700ms) · length (patient/eager by duration) · p_based
(floor + span*P(still speaking)). Runs headless, no mic. Real ASR for the P
semantic cue is opt-in via --with-asr (slower); default uses acoustic cues only.
"""
from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import soundfile as sf

from soca.asr.vad import SpeechDetector
from soca.core.endpoint import EndpointConfig, record_until_silence

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
    return EndpointConfig(adaptive=True, endpoint_mode=mode, max_record_ms=30000)


_DETECTOR: SpeechDetector | None = None


def _record(audio: np.ndarray, mode: str, asr_fn) -> np.ndarray:
    kwargs = {}
    # Only p_based reads partial_text; running ASR for fixed/length is pure waste.
    if asr_fn is not None and mode == "p_based":
        kwargs["on_partial"] = lambda c, t: None
        kwargs["partial_transcriber"] = asr_fn
    return record_until_silence(
        _DETECTOR,
        config=_config(mode),
        stream_factory=lambda **k: ArrayStream(audio, 1600),
        **kwargs,
    )


def run_false_cut(phrase, gap_ms, mode, noise, asr_fn) -> bool:
    """True = closed during the gap (before the 2nd half) = a false cut."""
    split = split_at_dip(phrase)
    part_a, part_b = phrase[:split], phrase[split:]
    audio = np.concatenate([
        part_a, _silence(gap_ms, noise), part_b, _silence(TAIL_S * 1000, noise)
    ])
    end_of_speech = len(part_a) + int(SR * gap_ms / 1000) + len(part_b)
    got = _record(audio, mode, asr_fn)
    return len(got) < end_of_speech


def run_end_latency(phrase, mode, noise, asr_fn) -> float | None:
    """ms from real speech end to close, on a clean phrase (no inserted gap)."""
    audio = np.concatenate([phrase, _silence(TAIL_S * 1000, noise)])
    got = _record(audio, mode, asr_fn)
    if len(got) < len(phrase):
        return None                                  # cut inside the phrase -> skip
    return (len(got) - len(phrase)) / (SR / 1000)


def build_asr_fn(enabled: bool):
    if not enabled:
        return None
    from soca.asr import VietnameseASR

    asr = VietnameseASR(num_threads=4)

    def transcribe(audio):
        return getattr(asr.transcribe(audio), "text", "") or ""

    return transcribe


@click.command()
@click.option("--n", default=24, help="number of FLEURS phrases")
@click.option("--gaps", default="300,500,700,900,1200", help="hesitation gaps (ms)")
@click.option("--noise", default=0.0, help="silence noise floor amplitude (0 = pure)")
@click.option("--with-asr", is_flag=True, help="run real ASR for the P semantic cue")
def main(n: int, gaps: str, noise: float, with_asr: bool) -> None:
    global _DETECTOR
    from local import config as cfg

    _DETECTOR = SpeechDetector()
    asr_fn = build_asr_fn(with_asr)
    gap_list = [int(g) for g in gaps.split(",")]
    modes = ("fixed", "length", "p_based")

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

    print(f"phrases={len(phrases)}  asr={'on' if asr_fn else 'off'}  noise={noise}\n")

    # --- Metric 1: false-cut rate vs gap ---
    print("false_cut_rate (lower = more patient, does NOT chop hesitation)")
    header = "gap_ms  | " + " | ".join(f"{m:>8}" for m in modes)
    print(header + "\n" + "-" * len(header))
    fc: dict[str, dict[int, float]] = {m: {} for m in modes}
    for g in gap_list:
        cells = []
        for m in modes:
            rate = sum(run_false_cut(ph, g, m, noise, asr_fn) for ph in phrases) / len(phrases)
            fc[m][g] = round(rate, 3)
            cells.append(f"{rate:7.0%} ")
        print(f"{g:6d}  | " + " | ".join(cells))

    # --- Metric 2: end-of-turn latency on clean phrases ---
    print("\nend_latency_ms (lower = snappier when the user IS done)")
    lat: dict[str, float | None] = {}
    for m in modes:
        vals = [run_end_latency(ph, m, noise, asr_fn) for ph in phrases]
        vals = [v for v in vals if v is not None]
        lat[m] = round(sum(vals) / len(vals), 0) if vals else None
        print(f"  {m:>8}: {lat[m]}ms  (n={len(vals)})")

    out = Path(cfg.EVAL_RESULTS_DIR) / "endpoint_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"phrases": len(phrases), "asr": bool(asr_fn), "noise": noise,
         "false_cut_rate": fc, "end_latency_ms": lat}, indent=2, ensure_ascii=False,
    ))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
