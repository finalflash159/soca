from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.valtec_torch_reference import synthesize_torch_reference
from soca.tts.valtec import (
    ValtecOnnxTTS,
    ValtecVietnameseFrontend,
    resolve_valtec_onnx_artifacts,
)

VOICES = ("NF", "SF", "NM1", "SM", "NM2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ParityMetrics:
    sample_rate_match: bool
    duration_ratio: float
    waveform_mae: float | None
    spectral_cosine: float


def _spectrum(audio: np.ndarray, bins: int = 2048) -> np.ndarray:
    if audio.size == 0:
        return np.zeros(bins // 2 + 1, dtype=np.float64)
    window = np.hanning(min(audio.size, bins))
    frame = np.zeros(bins, dtype=np.float64)
    frame[: window.size] = audio[: window.size] * window
    return np.log1p(np.abs(np.fft.rfft(frame)))


def compare_audio(
    torch_audio: np.ndarray,
    onnx_audio: np.ndarray,
    *,
    torch_sample_rate: int,
    onnx_sample_rate: int,
    same_checkpoint: bool,
) -> ParityMetrics:
    duration_ratio = len(onnx_audio) / max(len(torch_audio), 1)
    left, right = _spectrum(torch_audio), _spectrum(onnx_audio)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    spectral_cosine = float(np.dot(left, right) / denominator) if denominator else 0.0
    waveform_mae: float | None = None
    if same_checkpoint and len(torch_audio) == len(onnx_audio):
        waveform_mae = float(np.mean(np.abs(torch_audio - onnx_audio)))
    return ParityMetrics(
        sample_rate_match=torch_sample_rate == onnx_sample_rate,
        duration_ratio=duration_ratio,
        waveform_mae=waveform_mae,
        spectral_cosine=spectral_cosine,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Valtec Torch and ONNX inference.")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--allow-reference", action="store_true")
    parser.add_argument("--allow-candidate", action="store_true")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trust-checkpoint", action="store_true")
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", choices=VOICES, default="NF")
    parser.add_argument("--same-checkpoint", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifacts = resolve_valtec_onnx_artifacts(
        args.artifact_root,
        variant=args.variant,
        allow_reference=args.allow_reference,
        allow_candidate=args.allow_candidate,
    )
    frontend = ValtecVietnameseFrontend.from_artifacts(artifacts)
    model_inputs = frontend.prepare(args.text)
    speaker_id = artifacts.voice_map[args.voice]
    torch_result = synthesize_torch_reference(
        model_inputs,
        speaker_id=speaker_id,
        checkpoint=args.checkpoint,
        config=args.config,
        trust_checkpoint=args.trust_checkpoint,
    )
    onnx_engine = ValtecOnnxTTS(
        artifact_root=args.artifact_root,
        artifact_variant=args.variant,
        allow_reference=args.allow_reference,
        allow_candidate=args.allow_candidate,
        frontend=frontend,
        noise_scale=0.0,
        length_scale=1.0,
        seed=0,
        # The Torch reference renders the full text in one pass, so the ONNX
        # side must skip sentence chunking for a like-for-like comparison.
        sentence_chunking=False,
    )
    onnx_result = onnx_engine.synthesize(args.text, voice=args.voice)
    metrics = compare_audio(
        torch_result.audio,
        onnx_result.audio,
        torch_sample_rate=torch_result.sample_rate,
        onnx_sample_rate=onnx_result.sample_rate,
        same_checkpoint=args.same_checkpoint,
    )
    payload = {
        "voice": args.voice,
        "artifact_id": artifacts.artifact_id,
        "variant": artifacts.variant,
        "precision": artifacts.precision,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "config_sha256": _sha256(args.config),
        "frontend": onnx_engine.frontend_metadata,
        "metrics": asdict(metrics),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    sf.write(args.audio_dir / f"{args.voice}_torch.wav", torch_result.audio, torch_result.sample_rate)
    sf.write(args.audio_dir / f"{args.voice}_onnx.wav", onnx_result.audio, onnx_result.sample_rate)
    if not metrics.sample_rate_match or not 0.95 <= metrics.duration_ratio <= 1.05:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
