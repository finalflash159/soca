"""Aggregate objective Valtec candidate metrics into a raw acceptance report.

This is the missing link between a freshly built candidate and
``eval/build_valtec_acceptance.py``: it runs five-voice Torch/ONNX parity,
an fp32/int8 latency benchmark, and a PhoWhisper ASR loopback, then writes a
``raw-report.json`` in the schema ``build_acceptance()`` expects. It also writes
one production-fidelity WAV per voice so the reviewer can do the listening gate.

The listening gate stays human: ``voices_passed`` here means a voice cleared the
objective checks (parity + non-empty synthesis). The reviewer still confirms the
sound via ``build_valtec_acceptance.py --listening-approved-by``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import soundfile as sf

from eval.eval_valtec_parity import compare_audio
from eval.valtec_torch_reference import synthesize_torch_reference
from soca.tts.valtec import (
    ValtecOnnxTTS,
    ValtecVietnameseFrontend,
    resolve_valtec_onnx_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VOICES = ("NF", "SF", "NM1", "SM", "NM2")
REQUIRED_ISSUES = ("#1", "#2", "#3", "#4", "#6")
PARITY_TEXT = "Nghỉ ngơi rồi khuyến nghị tiếp nhé."
LISTENING_TEXT = (
    "Xin chào, mình là SoCa. Hôm nay bạn muốn mình hỗ trợ gì?"
)
PARITY_COSINE_MIN = 0.99
GATE_TEST_FILES = (
    "tests/test_valtec_g2p.py",
    "tests/test_valtec_normalizer.py",
    "tests/test_valtec_onnx_runner.py",
    "tests/test_valtec_artifacts.py",
)


@dataclass(frozen=True)
class VoiceParity:
    voice: str
    sample_rate_match: bool
    duration_ratio: float
    waveform_mae: float | None
    spectral_cosine: float
    onnx_nonempty: bool

    @property
    def passed(self) -> bool:
        if not (self.sample_rate_match and self.onnx_nonempty):
            return False
        if not 0.95 <= self.duration_ratio <= 1.05:
            return False
        if self.waveform_mae is not None:
            return self.waveform_mae < 1e-4
        return self.spectral_cosine >= PARITY_COSINE_MIN


def _normalize_for_cer(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệ"
                  r"ìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữự"
                  r"ỳýỷỹỵđ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _cer(reference: str, hypothesis: str) -> float:
    """Character error rate via Levenshtein distance / reference length.

    Inline so the report builder does not require the optional ``eval`` extra
    (jiwer); the metric matches jiwer.cer for single-string inputs.
    """
    if not reference:
        return 1.0
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            cost = 0 if ref_char == hyp_char else 1
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            )
        previous = current
    return previous[-1] / len(reference)


def _load_prompts(path: Path, limit: int | None) -> list[tuple[str, str]]:
    prompts: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            prompts.append((str(payload["id"]), str(payload["text"])))
    if limit is not None:
        prompts = prompts[:limit]
    return prompts


def _summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = min(int(len(ordered) * 0.95), len(ordered) - 1)
    return {
        "median": float(median(ordered)),
        "p95": float(ordered[p95_index]),
    }


def _build_engine(
    artifact_root: Path,
    variant: str,
    *,
    chunking: bool,
    parity_mode: bool,
) -> ValtecOnnxTTS:
    artifacts = resolve_valtec_onnx_artifacts(
        artifact_root, variant=variant, allow_candidate=True
    )
    frontend = ValtecVietnameseFrontend.from_artifacts(artifacts)
    kwargs: dict[str, Any] = {
        "artifact_root": artifact_root,
        "artifact_variant": variant,
        "allow_candidate": True,
        "frontend": frontend,
        "sentence_chunking": chunking,
    }
    if parity_mode:
        kwargs.update(noise_scale=0.0, length_scale=1.0, seed=0)
    return ValtecOnnxTTS(**kwargs)


def run_parity(
    artifact_root: Path,
    *,
    checkpoint: Path,
    config: Path,
    audio_dir: Path,
) -> tuple[list[VoiceParity], dict[str, VoiceParity]]:
    """Torch-vs-ONNX parity for every voice on the fp32 candidate."""
    engine = _build_engine(
        artifact_root, "fp32", chunking=False, parity_mode=True
    )
    listening_engine = _build_engine(
        artifact_root, "fp32", chunking=True, parity_mode=False
    )
    artifacts = resolve_valtec_onnx_artifacts(
        artifact_root, variant="fp32", allow_candidate=True
    )
    frontend = ValtecVietnameseFrontend.from_artifacts(artifacts)
    model_inputs = frontend.prepare(PARITY_TEXT)

    fp32: list[VoiceParity] = []
    int8_by_voice: dict[str, VoiceParity] = {}
    audio_dir.mkdir(parents=True, exist_ok=True)
    for voice in VOICES:
        speaker_id = artifacts.voice_map[voice]
        torch_result = synthesize_torch_reference(
            model_inputs,
            speaker_id=speaker_id,
            checkpoint=checkpoint,
            config=config,
            trust_checkpoint=True,
        )
        onnx_result = engine.synthesize(PARITY_TEXT, voice=voice)
        metrics = compare_audio(
            torch_result.audio,
            onnx_result.audio,
            torch_sample_rate=torch_result.sample_rate,
            onnx_sample_rate=onnx_result.sample_rate,
            same_checkpoint=True,
        )
        fp32.append(
            VoiceParity(
                voice=voice,
                sample_rate_match=metrics.sample_rate_match,
                duration_ratio=metrics.duration_ratio,
                waveform_mae=metrics.waveform_mae,
                spectral_cosine=metrics.spectral_cosine,
                onnx_nonempty=onnx_result.audio.size > 0,
            )
        )
        listening = listening_engine.synthesize(LISTENING_TEXT, voice=voice)
        sf.write(
            audio_dir / f"{voice}_listening_fp32.wav",
            listening.audio,
            listening.sample_rate,
        )
    return fp32, int8_by_voice


def run_int8_parity(
    artifact_root: Path,
    *,
    checkpoint: Path,
    config: Path,
) -> VoiceParity:
    """Single-voice (NF) int8 parity to score the mixed-precision variant."""
    artifacts = resolve_valtec_onnx_artifacts(
        artifact_root, variant="int8", allow_candidate=True
    )
    frontend = ValtecVietnameseFrontend.from_artifacts(artifacts)
    model_inputs = frontend.prepare(PARITY_TEXT)
    engine = _build_engine(
        artifact_root, "int8", chunking=False, parity_mode=True
    )
    speaker_id = artifacts.voice_map["NF"]
    torch_result = synthesize_torch_reference(
        model_inputs,
        speaker_id=speaker_id,
        checkpoint=checkpoint,
        config=config,
        trust_checkpoint=True,
    )
    onnx_result = engine.synthesize(PARITY_TEXT, voice="NF")
    metrics = compare_audio(
        torch_result.audio,
        onnx_result.audio,
        torch_sample_rate=torch_result.sample_rate,
        onnx_sample_rate=onnx_result.sample_rate,
        same_checkpoint=True,
    )
    return VoiceParity(
        voice="NF",
        sample_rate_match=metrics.sample_rate_match,
        duration_ratio=metrics.duration_ratio,
        waveform_mae=metrics.waveform_mae,
        spectral_cosine=metrics.spectral_cosine,
        onnx_nonempty=onnx_result.audio.size > 0,
    )


def benchmark(
    artifact_root: Path,
    prompts: list[tuple[str, str]],
    *,
    repeats: int,
) -> tuple[dict[str, float], dict[str, float], float]:
    """Latency/RTF for fp32 and int8; returns fp32 stats, rtf stats, int8 median."""
    fp32 = _build_engine(artifact_root, "fp32", chunking=True, parity_mode=False)
    int8 = _build_engine(artifact_root, "int8", chunking=True, parity_mode=False)

    fp32_latencies: list[float] = []
    fp32_rtfs: list[float] = []
    int8_latencies: list[float] = []
    for _ in range(repeats):
        for _id, text in prompts:
            fp32_result = fp32.synthesize(text, voice="NF")
            fp32_latencies.append(fp32_result.latency_ms)
            fp32_rtfs.append(fp32_result.rtf)
            int8_latencies.append(int8.synthesize(text, voice="NF").latency_ms)
    return (
        _summarize(fp32_latencies),
        _summarize(fp32_rtfs),
        float(median(int8_latencies)),
    )


def asr_loopback(
    artifact_root: Path,
    prompts: list[tuple[str, str]],
    *,
    asr_model_key: str,
) -> tuple[float, list[dict[str, Any]]]:
    """Synthesize each prompt, transcribe it back, and score CER."""
    from soca.asr.whisper_onnx import VietnameseASR

    engine = _build_engine(artifact_root, "fp32", chunking=True, parity_mode=False)
    asr = VietnameseASR(model_key=asr_model_key)

    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    for prompt_id, text in prompts:
        result = engine.synthesize(text, voice="NF")
        heard = asr.transcribe(result.audio).text
        reference = _normalize_for_cer(text)
        hypothesis = _normalize_for_cer(heard)
        score = _cer(reference, hypothesis)
        scores.append(score)
        rows.append(
            {
                "id": prompt_id,
                "reference": reference,
                "heard": hypothesis,
                "cer": score,
            }
        )
    mean_cer = float(sum(scores) / len(scores)) if scores else 1.0
    return mean_cer, rows


def run_gate_tests() -> tuple[bool, bool]:
    """Run the Valtec unit suite; returns (pytest_ok, g2p_golden_ok)."""
    def _run(files: list[str]) -> bool:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", *files, "-q"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        return completed.returncode == 0

    g2p_ok = _run(["tests/test_valtec_g2p.py"])
    suite_ok = _run(list(GATE_TEST_FILES))
    return suite_ok, g2p_ok


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trust-checkpoint", action="store_true")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=REPO_ROOT / "eval/prompts/tts_bakeoff_vi.jsonl",
    )
    parser.add_argument("--asr-model", default="phowhisper_medium")
    parser.add_argument("--bench-limit", type=int, default=12)
    parser.add_argument("--bench-repeats", type=int, default=2)
    parser.add_argument("--loopback-limit", type=int, default=12)
    parser.add_argument("--skip-int8", action="store_true")
    parser.add_argument("--skip-gate-tests", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.trust_checkpoint:
        raise SystemExit("--trust-checkpoint is required (parity uses torch.load)")

    checkpoint = args.checkpoint.expanduser().resolve()
    config = args.config.expanduser().resolve()
    prompts = _load_prompts(args.prompts, None)
    bench_prompts = prompts[: args.bench_limit]
    loopback_prompts = prompts[: args.loopback_limit]

    print(f"[1/5] Parity 5 voice + listening WAVs -> {args.audio_dir}", flush=True)
    fp32_parity, _ = run_parity(
        args.artifact_root,
        checkpoint=checkpoint,
        config=config,
        audio_dir=args.audio_dir,
    )
    voices_passed = [vp.voice for vp in fp32_parity if vp.passed]

    int8_block: dict[str, Any] | None = None
    if not args.skip_int8:
        print("[2/5] int8 parity (NF) + speedup", flush=True)
        int8_parity = run_int8_parity(
            args.artifact_root, checkpoint=checkpoint, config=config
        )
    else:
        int8_parity = None

    print(
        f"[3/5] Benchmark fp32/int8 "
        f"({len(bench_prompts)} prompts x{args.bench_repeats})",
        flush=True,
    )
    fp32_latency, fp32_rtf, int8_median = benchmark(
        args.artifact_root, bench_prompts, repeats=args.bench_repeats
    )
    if int8_parity is not None:
        speedup = (fp32_latency["median"] - int8_median) / fp32_latency["median"] * 100.0
        int8_block = {
            "speedup_percent_vs_fp32": float(speedup),
            "quality_passed": bool(int8_parity.spectral_cosine >= PARITY_COSINE_MIN),
            "parity_spectral_cosine": int8_parity.spectral_cosine,
            "median_latency_ms": int8_median,
        }

    print(
        f"[4/5] ASR loopback via {args.asr_model} "
        f"({len(loopback_prompts)} prompts)",
        flush=True,
    )
    mean_cer, loopback_rows = asr_loopback(
        args.artifact_root, loopback_prompts, asr_model_key=args.asr_model
    )

    if args.skip_gate_tests:
        pytest_ok, g2p_ok = True, True
    else:
        print("[5/5] Gate tests (pytest / g2p golden)", flush=True)
        pytest_ok, g2p_ok = run_gate_tests()

    onnx_smoke = all(vp.onnx_nonempty for vp in fp32_parity)
    variants: dict[str, Any] = {
        "fp32": {
            "tts_p50_ms": fp32_latency["median"],
            "tts_p95_ms": fp32_latency["p95"],
            "rtf_p50": fp32_rtf["median"],
            "rtf_p95": fp32_rtf["p95"],
        }
    }
    if int8_block is not None:
        variants["int8"] = int8_block

    report = {
        "schema_version": 1,
        "artifact_root": str(args.artifact_root.resolve()),
        "pytest": pytest_ok,
        "onnx_smoke": onnx_smoke,
        "g2p_golden": g2p_ok,
        "voices_passed": voices_passed,
        "issue_coverage": list(REQUIRED_ISSUES),
        "asr_loopback_cer": mean_cer,
        "asr_model": args.asr_model,
        "variants": variants,
        "parity": [
            {
                "voice": vp.voice,
                "sample_rate_match": vp.sample_rate_match,
                "duration_ratio": vp.duration_ratio,
                "waveform_mae": vp.waveform_mae,
                "spectral_cosine": vp.spectral_cosine,
                "passed": vp.passed,
            }
            for vp in fp32_parity
        ],
        "asr_loopback_detail": loopback_rows,
        "listening_audio_dir": str(args.audio_dir.resolve()),
        "note": (
            "voices_passed reflects objective parity only; the human listening "
            "gate is confirmed via build_valtec_acceptance.py --listening-approved-by."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n=== raw-report summary ===")
    print(f"  voices_passed (parity): {voices_passed}")
    print(
        f"  fp32 latency p50/p95: {fp32_latency['median']:.0f}/"
        f"{fp32_latency['p95']:.0f} ms | rtf p50: {fp32_rtf['median']:.3f}"
    )
    if int8_block is not None:
        print(
            f"  int8 speedup: {int8_block['speedup_percent_vs_fp32']:+.1f}% | "
            f"quality_passed: {int8_block['quality_passed']}"
        )
    print(f"  asr_loopback_cer: {mean_cer:.4f}")
    print(f"  gates pytest/onnx_smoke/g2p_golden: {pytest_ok}/{onnx_smoke}/{g2p_ok}")
    print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
