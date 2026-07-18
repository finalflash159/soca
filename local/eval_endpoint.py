"""Eval endpoint: fixed-700 vs adaptive on audio with synthetic "hesitation".

Insert gap_ms of silence in the MIDDLE of speech (simulated hesitation), pad 1.5s tail,
feed via ArrayStream. early_cut = returned audio SHORTER than the true end-of-speech. close_latency
= (returned audio - end-of-speech) in ms. Runs headless, NO mic needed.
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
TAIL_S = 1.5


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
        if len(chunk) < n:                       # exhausted -> pad with silence
            chunk = np.pad(chunk, (0, n - len(chunk)))
        return chunk.reshape(-1, 1), False


def synth_hesitation(audio: np.ndarray, gap_ms: int) -> np.ndarray:
    """Insert a mid gap + pad tail. Returns new audio; end-of-speech = len - tail."""
    mid = len(audio) // 2
    gap = np.zeros(int(SR * gap_ms / 1000), dtype=np.float32)
    tail = np.zeros(int(SR * TAIL_S), dtype=np.float32)
    return np.concatenate([audio[:mid], gap, audio[mid:], tail])


def load_manifest(n: int) -> list[Path]:
    """Read the first n wav paths from the FLEURS manifest (like local/eval_table7.py).

    Adjust the path for your machine; keep it light so it runs fast.
    """
    from local import config as cfg

    root = Path(cfg.FLEURS_WAV_DIR)
    wavs = sorted(root.glob("*.wav"))[:n]
    if not wavs:
        raise SystemExit(f"No wav found in {root} — fix local/config.py")
    return wavs


def run_one(path: Path, gap_ms: int, adaptive: bool, detector: SpeechDetector) -> dict:
    audio, _sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)               # to mono
    audio = synth_hesitation(audio.astype(np.float32), gap_ms)
    speech_end = len(audio) - int(SR * TAIL_S)   # the REAL end-of-speech point
    config = EndpointConfig(adaptive=adaptive, max_record_ms=30000)
    got = record_until_silence(
        detector, config=config, stream_factory=lambda **kw: ArrayStream(audio, 1600)
    )
    return {
        "early_cut": len(got) < speech_end,                    # closed before speech ended
        "close_latency_ms": max(0, len(got) - speech_end) / (SR / 1000),
    }


@click.command()
@click.option("--n", default=20, help="number of FLEURS samples")
@click.option("--gap-ms", default=800, help="hesitation length inserted mid-sentence")
def main(n: int, gap_ms: int) -> None:
    detector = SpeechDetector()
    paths = load_manifest(n)
    summary: dict[str, dict] = {}
    for label, adaptive in (("fixed", False), ("adaptive", True)):
        rows = [run_one(p, gap_ms, adaptive, detector) for p in paths]
        cuts = sum(r["early_cut"] for r in rows)
        lat = [r["close_latency_ms"] for r in rows if not r["early_cut"]]
        summary[label] = {
            "n": len(rows),
            "early_cut_rate": round(cuts / len(rows), 3),
            "close_latency_ms_mean": round(sum(lat) / len(lat), 1) if lat else None,
        }
        print(
            f"{label:8s} gap={gap_ms}ms  early_cut={summary[label]['early_cut_rate']:.0%}  "
            f"close_latency≈{summary[label]['close_latency_ms_mean']}ms"
        )
    Path("eval/results").mkdir(parents=True, exist_ok=True)
    out = Path("eval/results/endpoint_eval.json")
    out.write_text(json.dumps({"gap_ms": gap_ms, "summary": summary}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
