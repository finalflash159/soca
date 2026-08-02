from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.tts_intelligibility.corpora import build_all_corpora  # noqa: E402
from eval.tts_intelligibility.manifest import (  # noqa: E402
    SynthManifest,
    read_manifest,
    record_from_item,
    write_manifest,
)
from eval.tts_intelligibility.scoring import aggregate, score_item  # noqa: E402

DEFAULT_OUTPUT_DIR = REPO_ROOT / "eval" / "results" / "tts_intelligibility"
ASR_SAMPLE_RATE = 16_000


def _resample_to_asr(audio: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == ASR_SAMPLE_RATE:
        return audio.astype(np.float32, copy=False)
    import math

    from scipy.signal import resample_poly

    divisor = math.gcd(source_rate, ASR_SAMPLE_RATE)
    return resample_poly(
        audio, ASR_SAMPLE_RATE // divisor, source_rate // divisor
    ).astype(np.float32, copy=False)


def run_synth(args: argparse.Namespace) -> int:
    from soca.tts import TTSRuntimeUnavailableError, create_tts_engine

    output_dir = Path(args.output_dir)
    wav_dir = output_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    try:
        engine = create_tts_engine(voice=args.voice)
    except TTSRuntimeUnavailableError as exc:
        print(f"TTS engine unavailable: {exc}", file=sys.stderr)
        return 2

    # Valtec seeds its sampler from the OS when no seed is given, so the same
    # sentence yields different audio on every run and the score moves with it.
    # Two runs of the control corpus differed by 3 of 8 items before this was
    # pinned, which is larger than the engine differences this harness exists to
    # detect. Set it where the engine exposes it and say so when it does not.
    if getattr(engine, "_rng", None) is not None:
        setattr(engine, "_rng", np.random.default_rng(args.seed))  # noqa: B010
        print(f"Pinned TTS sampler seed to {args.seed}.")
    else:
        print(
            "WARNING: this engine exposes no sampler seed; synthesis may not be "
            "reproducible and scores will move between runs.",
            file=sys.stderr,
        )

    corpora = build_all_corpora(lexicon_limit=args.lexicon_limit)
    records = []
    total = sum(len(items) for items in corpora.values())
    done = 0

    for corpus_name, items in corpora.items():
        for item in items:
            result = engine.synthesize(item.text_in)
            wav_path = wav_dir / f"{item.item_id}.wav"
            sf.write(wav_path, result.audio, result.sample_rate)
            records.append(
                record_from_item(
                    item,
                    wav_path=wav_path,
                    sample_rate=result.sample_rate,
                    tts_latency_ms=result.latency_ms,
                    audio_duration_ms=result.audio_duration_ms,
                )
            )
            done += 1
            print(
                f"[{done:3d}/{total}] {corpus_name:<10} {item.item_id}  "
                f"{result.latency_ms:6.0f}ms",
                flush=True,
            )

    manifest = SynthManifest(
        engine=getattr(engine, "ENGINE_NAME", "unknown"),
        voice=args.voice or "",
        records=tuple(records),
    )
    manifest_path = output_dir / "synth_manifest.json"
    write_manifest(manifest, manifest_path)
    print(f"\nWrote {len(records)} utterances -> {manifest_path}")
    return 0


def _build_asr(args: argparse.Namespace):
    """Build the recognizer used to listen back to the synthesized audio.

    The lexicon corpus is English technical vocabulary, which a Vietnamese-only
    recognizer cannot return verbatim no matter how well the TTS said it. Such
    a recognizer therefore measures itself rather than the engine under test,
    so a code-switching backend is required for that corpus.
    """
    if args.asr_backend == "qwen":
        from soca.asr.qwen_backend import QwenASRBackend

        return QwenASRBackend(
            Path(args.qwen_model_path).expanduser().resolve(),
            # An empty context is deliberate: a prompted recognizer can repair a
            # mispronunciation from context and hide the very defect being
            # measured, so it is given no help.
            context=args.qwen_context,
            require_logprob=False,
        )

    from soca.asr.whisper_onnx import VietnameseASR

    return VietnameseASR(model_key=args.asr_model)


def run_score(args: argparse.Namespace) -> int:
    manifest = read_manifest(Path(args.manifest))
    asr = _build_asr(args)
    if args.corpus:
        manifest = SynthManifest(
            engine=manifest.engine,
            voice=manifest.voice,
            records=tuple(r for r in manifest.records if r.corpus in args.corpus),
        )

    verdicts = []
    for index, record in enumerate(manifest.records, start=1):
        audio, sample_rate = sf.read(record.wav_path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        asr_result = asr.transcribe(_resample_to_asr(audio, sample_rate))
        verdict = score_item(
            item_id=record.item_id,
            corpus=record.corpus,
            text_in=record.text_in,
            expected=record.expected,
            heard=asr_result.text,
            mode=record.mode,
            avg_logprob=asr_result.avg_logprob,
        )
        verdicts.append(verdict)
        mark = "ok  " if verdict.passed else "FAIL"
        print(
            f"[{index:3d}/{len(manifest.records)}] {mark} {record.item_id:<22} "
            f"heard={asr_result.text[:60]!r}",
            flush=True,
        )

    summaries = aggregate(verdicts)
    asr_label = (
        f"qwen:{args.qwen_model_path}" if args.asr_backend == "qwen" else args.asr_model
    )
    print(f"\n=== TTS intelligibility: engine={manifest.engine} voice={manifest.voice} ===")
    print(f"ASR: {asr_label}\n")
    print(f"{'corpus':<12} {'pass':>10} {'rate':>8} {'mean WER':>10}")
    for name in ("control", "normalizer", "lexicon"):
        summary = summaries.get(name)
        if summary is None:
            continue
        print(
            f"{name:<12} {summary.passed:>4}/{summary.total:<5} "
            f"{summary.pass_rate:>7.1%} {summary.mean_wer:>10.3f}"
        )

    control = summaries.get("control")
    if control is not None:
        # The control corpus is plain Vietnamese both engines should handle, so
        # whatever it loses is lost by the measuring chain itself (TTS voice,
        # resampling, ASR), not by the terms under test. Its WER is therefore
        # the resolution limit: a gap smaller than this between two engines is
        # not evidence of anything. Exact-match pass rate is reported too but is
        # deliberately not the gate -- one dropped article fails it while
        # costing almost no WER.
        print(f"\nMeasuring-chain noise floor (control WER): {control.mean_wer:.3f}")
        print(
            f"  -> differences below ~{control.mean_wer:.2f} WER between engines "
            "are not resolvable with this chain."
        )
        if control.mean_wer > 0.15:
            print(
                "  -> WARNING: noise floor is high. Improve the chain (stronger "
                "ASR, check the voice) before trusting the corpus rows above."
            )

    for name in ("normalizer", "lexicon", "control"):
        summary = summaries.get(name)
        if summary is None or not summary.failures:
            continue
        print(f"\n--- {name} failures ({len(summary.failures)}) ---")
        for verdict in summary.failures[: args.max_failures]:
            print(f"  expected {verdict.expected!r}")
            print(f"  heard    {verdict.heard!r}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether TTS output survives a TTS -> ASR round trip. "
            "Stage 1 synthesizes to WAV; stage 2 transcribes and scores, so the "
            "ASR may run in a different interpreter than the TTS."
        )
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    synth = sub.add_parser("synth", help="Synthesize every corpus item to WAV.")
    synth.add_argument("--voice", default=None)
    synth.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    synth.add_argument(
        "--lexicon-limit",
        type=int,
        default=None,
        help="Cap the lexicon corpus; omit to synthesize all curated terms.",
    )
    synth.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Seed for the TTS sampler so a re-run reproduces the same audio.",
    )
    synth.set_defaults(func=run_synth)

    score = sub.add_parser("score", help="Transcribe the WAVs and report.")
    score.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "synth_manifest.json",
    )
    score.add_argument("--asr-model", default="phowhisper_small")
    score.add_argument(
        "--asr-backend",
        choices=("phowhisper", "qwen"),
        default="phowhisper",
        help=(
            "Recognizer family. The lexicon corpus needs 'qwen': a "
            "Vietnamese-only recognizer cannot return English terms verbatim "
            "and would score the recognizer instead of the engine."
        ),
    )
    score.add_argument("--qwen-model-path", default="")
    score.add_argument("--qwen-context", default="")
    score.add_argument(
        "--corpus",
        action="append",
        default=[],
        help="Restrict scoring to a corpus (repeatable).",
    )
    score.add_argument("--max-failures", type=int, default=15)
    score.set_defaults(func=run_score)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
