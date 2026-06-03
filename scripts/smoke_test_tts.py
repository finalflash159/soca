"""Smoke test one local TTS engine and optionally write WAV files."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import soundfile as sf
from rich.console import Console
from rich.table import Table

from soca.core.audio_out import SoundDevicePlayer
from soca.tts import (
    DEFAULT_TTS_MODEL_KEY,
    TTS_MODEL_REGISTRY,
    TTSRuntimeUnavailableError,
    create_tts_engine,
)

console = Console(width=180)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "eval" / "results" / "tts_smoke"
DEFAULT_SENTENCES_PER_VOICE = 3

VOICE_TEST_SUITES = {
    "smoke": [
        "Bữa sau tập nên có một nguồn đạm, rau xanh và tinh bột vừa đủ.",
        "Lịch check-in là 7 giờ 30 tối, ngày 29 tháng 5 năm 2026.",
    ],
    "full": [
        "Xin chào, tôi là SoCa.",
        "Bạn muốn mình hỗ trợ gì hôm nay?",
        "Mình sẽ tạo checklist hôm nay cho bạn.",
        "Nếu thấy đau nhói hoặc chóng mặt, hãy dừng buổi tập và nghỉ vài phút.",
        "Bữa sau tập nên có một nguồn đạm, rau xanh và tinh bột vừa đủ.",
        "Cân nặng một ngày có thể dao động vì nước, muối và giấc ngủ.",
        "Mục tiêu tuần này là 150 phút vận động mức vừa.",
        "Lịch check-in là 7 giờ 30 tối, ngày 29 tháng 5 năm 2026.",
        "Protein có thể nằm trong khoảng 1.4 đến 2.0 gam trên mỗi ki-lô-gam mỗi ngày.",
        "Wi-Fi yếu thì bạn nên thử lại sau, hoặc chuyển sang mạng ổn định hơn.",
        "Bạn có thể bắt đầu bằng squat, push-up, row và plank, mỗi bài hai đến ba hiệp.",
        "Nếu không chắc chắn về một triệu chứng sức khỏe, hãy hỏi bác sĩ hoặc chuyên gia phù hợp.",
    ],
}


def safe_filename(value: str) -> str:
    keep = [ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value]
    return "".join(keep).strip("_") or "tts"


def select_voices(
    available_voices: Sequence[str],
    requested_voices: Sequence[str],
    max_voices: int | None,
    default_voices: Sequence[str] | None = None,
) -> list[str]:
    selected_voices = list(requested_voices or default_voices or available_voices)
    if max_voices is not None:
        selected_voices = selected_voices[:max_voices]

    unknown_voices = sorted(set(selected_voices) - set(available_voices))
    if unknown_voices:
        raise ValueError(f"Unknown voice(s): {unknown_voices}. Available voices: {available_voices}")
    return selected_voices


def build_voice_texts(
    voice: str,
    suite: str,
    custom_texts: Sequence[str],
    sentences_per_voice: int = DEFAULT_SENTENCES_PER_VOICE,
) -> list[str]:
    spoken_voice_name = voice.replace("_", " ").replace("-", " ")
    intro = f"Đây là giọng {spoken_voice_name}."
    if custom_texts:
        texts = [intro, *custom_texts]
    else:
        texts = [intro, *VOICE_TEST_SUITES[suite]]
    if sentences_per_voice > 0:
        return texts[:sentences_per_voice]
    return texts


def build_voice_jobs(
    voice: str,
    suite: str,
    custom_texts: Sequence[str],
    sentences_per_voice: int = DEFAULT_SENTENCES_PER_VOICE,
    split_sentences: bool = False,
) -> list[str]:
    texts = build_voice_texts(
        voice,
        suite,
        custom_texts,
        sentences_per_voice=sentences_per_voice,
    )
    if split_sentences:
        return texts
    return [" ".join(texts)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a quick SoCa TTS smoke test.")
    parser.add_argument(
        "--model",
        default=DEFAULT_TTS_MODEL_KEY,
        choices=sorted(TTS_MODEL_REGISTRY),
        help="TTS registry key to load.",
    )
    parser.add_argument(
        "--voice",
        action="append",
        default=[],
        help=(
            "Voice/speaker id where supported. Can be passed multiple times. "
            "Defaults to every voice exposed by the engine."
        ),
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help=(
            "Optional text override after the voice intro. Can be passed multiple times. "
            "Defaults to the selected test suite."
        ),
    )
    parser.add_argument(
        "--suite",
        choices=sorted(VOICE_TEST_SUITES),
        default="smoke",
        help=(
            "Built-in text suite. 'smoke' is short enough to run across every voice; "
            "'full' is for deeper listening checks."
        ),
    )
    parser.add_argument(
        "--max-voices",
        type=int,
        default=None,
        help="Limit how many voices to test. Useful for Piper's large speaker list.",
    )
    parser.add_argument(
        "--sentences-per-voice",
        type=int,
        default=DEFAULT_SENTENCES_PER_VOICE,
        help=(
            "Total sentences to synthesize per voice, including the voice intro. "
            "Use 0 to run the whole selected text suite."
        ),
    )
    parser.add_argument(
        "--split-sentences",
        action="store_true",
        help=(
            "Synthesize each sentence separately. Default is one clip per voice "
            "so zero-shot voice-design models keep a more stable timbre."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-write", action="store_true", help="Do not write WAV files.")
    parser.add_argument("--play", action="store_true", help="Play synthesized audio through speakers.")
    parser.add_argument(
        "--output-sample-rate",
        type=int,
        default=None,
        help="Optional playback sample rate. Defaults to each model's sample rate.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    console.print(f"[bold]Loading TTS:[/bold] {args.model}")
    try:
        engine = create_tts_engine(args.model, voice=args.voice[0] if args.voice else None)
    except TTSRuntimeUnavailableError as exc:
        console.print(f"[yellow]Runtime unavailable:[/yellow] {exc}")
        return 2

    available_voices = engine.list_voices()
    default_voices = TTS_MODEL_REGISTRY[args.model].smoke_test_voices
    selected_voices = select_voices(
        available_voices,
        args.voice,
        args.max_voices,
        default_voices=default_voices,
    )
    sentences_per_voice = len(
        build_voice_texts(
            selected_voices[0],
            args.suite,
            args.text,
            sentences_per_voice=args.sentences_per_voice,
        )
    )
    jobs_per_voice = len(
        build_voice_jobs(
            selected_voices[0],
            args.suite,
            args.text,
            sentences_per_voice=args.sentences_per_voice,
            split_sentences=args.split_sentences,
        )
    )
    console.print(f"[green]✓ Model loaded[/green] voices={available_voices}\n")
    console.print(f"[bold]Testing voices:[/bold] {selected_voices}\n")
    console.print(
        f"[bold]Plan:[/bold] {len(selected_voices)} voice(s) × "
        f"{sentences_per_voice} sentence(s), grouped into {jobs_per_voice} synthesis call(s) "
        f"per voice = {len(selected_voices) * jobs_per_voice} total call(s)\n"
    )

    if not args.no_write:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    player = SoundDevicePlayer(output_sample_rate=args.output_sample_rate) if args.play else None

    table = Table(title=f"TTS Smoke Test — {args.model}", show_lines=True)
    table.add_column("Text", style="cyan", width=72, overflow="fold")
    table.add_column("Voice", justify="right")
    table.add_column("sr", justify="right")
    table.add_column("Dur ms", justify="right")
    table.add_column("Lat ms", justify="right")
    table.add_column("RTF", justify="right")
    table.add_column("Samples", justify="right")
    if args.play:
        table.add_column("Played", justify="right")
    if not args.no_write:
        table.add_column("Output", overflow="fold")

    for selected_voice in selected_voices:
        texts = build_voice_jobs(
            selected_voice,
            args.suite,
            args.text,
            sentences_per_voice=args.sentences_per_voice,
            split_sentences=args.split_sentences,
        )
        for text_index, text in enumerate(texts, start=1):
            result = engine.synthesize(text, voice=selected_voice)
            playback_status = ""
            if player is not None:
                playback = player.play(result.audio, result.sample_rate, blocking=True)
                playback_status = "yes" if playback.played else playback.reason

            output_path = ""
            if not args.no_write:
                output_path_obj = (
                    args.output_dir
                    / f"{safe_filename(args.model)}_{safe_filename(selected_voice)}_{text_index:02d}.wav"
                )
                sf.write(output_path_obj, result.audio, result.sample_rate)
                output_path = str(output_path_obj)

            row = [
                text,
                result.voice,
                str(result.sample_rate),
                f"{result.audio_duration_ms:.0f}",
                f"{result.latency_ms:.0f}",
                f"{result.rtf:.2f}",
                str(len(result.audio)),
            ]
            if args.play:
                row.append(playback_status)
            if not args.no_write:
                row.append(output_path)
            table.add_row(*row)

    console.print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
