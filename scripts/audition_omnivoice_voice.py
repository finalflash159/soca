"""Generate OmniVoice audition samples for picking a stable saved voice."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import soundfile as sf
from rich.console import Console
from rich.progress import track
from rich.table import Table

from soca.core.audio_out import SoundDevicePlayer
from soca.tts import TTS_MODEL_REGISTRY, create_tts_engine

console = Console(width=180)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "eval" / "results" / "omnivoice_auditions"
DEFAULT_REFERENCE_TEXT = (
    "Xin chào, tôi là Sơn Ca. "
    "Hôm nay mình sẽ giúp bạn theo dõi lịch tập, bữa ăn và các mục tiêu sức khỏe. "
    "Khi cần, mình sẽ nhắc bạn uống nước, ghi lại cân nặng và điều chỉnh kế hoạch nhẹ nhàng."
)


def safe_filename(value: str) -> str:
    keep = [ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value]
    return "".join(keep).strip("_") or "tts"


def select_voices(
    available_voices: Sequence[str],
    requested_voices: Sequence[str],
    default_voices: Sequence[str] | None,
    max_voices: int | None,
) -> list[str]:
    selected = list(requested_voices or default_voices or available_voices)
    if max_voices is not None:
        selected = selected[:max_voices]
    unknown = sorted(set(selected) - set(available_voices))
    if unknown:
        raise ValueError(f"Unknown OmniVoice voice(s): {unknown}. Available voices: {available_voices}")
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate multiple OmniVoice samples per voice-design preset. "
            "Listen to the WAV files, pick the best sample, then freeze it with "
            "scripts/freeze_omnivoice_voice.py."
        )
    )
    parser.add_argument("--samples-per-voice", type=int, default=10)
    parser.add_argument("--voice", action="append", default=[], help="Voice preset to audition. Repeatable.")
    parser.add_argument("--max-voices", type=int, default=None)
    parser.add_argument("--text", default=DEFAULT_REFERENCE_TEXT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--play", action="store_true", help="Play each generated sample while writing it.")
    parser.add_argument("--output-sample-rate", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.samples_per_voice < 1:
        raise ValueError("--samples-per-voice must be >= 1")

    config = TTS_MODEL_REGISTRY["omnivoice"]
    engine = create_tts_engine("omnivoice")
    available_voices = engine.list_voices()
    selected_voices = select_voices(
        available_voices,
        args.voice,
        config.smoke_test_voices,
        args.max_voices,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    player = SoundDevicePlayer(output_sample_rate=args.output_sample_rate) if args.play else None

    plan = len(selected_voices) * args.samples_per_voice
    console.print(f"[bold]OmniVoice audition[/bold] -> {output_dir}")
    console.print(f"[bold]Voices:[/bold] {selected_voices}")
    console.print(f"[bold]Plan:[/bold] {len(selected_voices)} voice(s) × {args.samples_per_voice} sample(s) = {plan}")
    console.print(f"[bold]Reference text:[/bold] {args.text}\n")

    rows: list[dict[str, object]] = []
    progress_items = [(voice, index) for voice in selected_voices for index in range(1, args.samples_per_voice + 1)]
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for voice, sample_index in track(progress_items, description="Generating audition samples..."):
            sample_id = f"{safe_filename(voice)}_{sample_index:02d}"
            voice_dir = output_dir / safe_filename(voice)
            voice_dir.mkdir(parents=True, exist_ok=True)
            wav_path = voice_dir / f"{sample_id}.wav"

            result = engine.synthesize(args.text, voice=voice)
            sf.write(wav_path, result.audio, result.sample_rate)
            if player is not None:
                player.play(result.audio, result.sample_rate, blocking=True)

            row = {
                "sample_id": sample_id,
                "voice": voice,
                "voice_instruction": engine._instruction_for_voice(voice),
                "text": args.text,
                "audio_path": str(wav_path.resolve()),
                "sample_rate": result.sample_rate,
                "duration_ms": result.audio_duration_ms,
                "latency_ms": result.latency_ms,
                "rtf": result.rtf,
            }
            rows.append(row)
            manifest_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model": "omnivoice",
                "voices": selected_voices,
                "samples_per_voice": args.samples_per_voice,
                "text": args.text,
                "manifest_jsonl": str(manifest_path.resolve()),
                "items": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    table = Table(title="OmniVoice Audition Samples", show_lines=True)
    table.add_column("Voice")
    table.add_column("Sample")
    table.add_column("Dur ms", justify="right")
    table.add_column("Lat ms", justify="right")
    table.add_column("RTF", justify="right")
    table.add_column("Path", overflow="fold")
    for row in rows:
        table.add_row(
            str(row["voice"]),
            str(row["sample_id"]),
            f"{float(row['duration_ms']):.0f}",
            f"{float(row['latency_ms']):.0f}",
            f"{float(row['rtf']):.2f}",
            str(row["audio_path"]),
        )
    console.print(table)
    console.print(f"\n[green]Saved manifest:[/green] {manifest_path}")
    console.print("[bold]Next:[/bold] pick a WAV you like, then run scripts/freeze_omnivoice_voice.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
