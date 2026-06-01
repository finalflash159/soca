"""Freeze one OmniVoice audition sample into a reusable saved voice."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console

from soca.tts import OmniVoiceTTS, create_tts_engine

console = Console()


def iter_manifest_rows(path: Path) -> list[dict[str, object]]:
    path = path.expanduser().resolve()
    if path.suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError(f"Manifest JSON does not contain an items list: {path}")
    return [item for item in items if isinstance(item, dict)]


def find_manifest(sample_path: Path) -> Path | None:
    for parent in [sample_path.parent, *sample_path.parents]:
        for name in ("manifest.jsonl", "manifest.json"):
            candidate = parent / name
            if candidate.exists():
                return candidate
    return None


def find_sample_row(sample_path: Path, manifest_path: Path | None) -> dict[str, object] | None:
    manifest_path = manifest_path or find_manifest(sample_path)
    if manifest_path is None:
        return None

    sample_resolved = sample_path.expanduser().resolve()
    for row in iter_manifest_rows(manifest_path):
        audio_path = row.get("audio_path")
        if not audio_path:
            continue
        try:
            if Path(str(audio_path)).expanduser().resolve() == sample_resolved:
                return row
        except FileNotFoundError:
            if Path(str(audio_path)).name == sample_path.name:
                return row
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Turn one audition WAV into a persistent OmniVoice voice_clone_prompt. "
            "The reference transcript is read from the audition manifest unless --ref-text is provided."
        )
    )
    parser.add_argument("--sample", type=Path, required=True, help="WAV sample chosen from audition output.")
    parser.add_argument("--voice-id", required=True, help="Saved voice id, e.g. sonca_female_01.")
    parser.add_argument("--ref-text", default=None, help="Exact transcript of --sample. Overrides manifest text.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional manifest.jsonl or manifest.json path.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sample_path = args.sample.expanduser().resolve()
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample not found: {sample_path}")

    row = find_sample_row(sample_path, args.manifest)
    ref_text = (args.ref_text or (str(row["text"]) if row and row.get("text") else "")).strip()
    if not ref_text:
        raise ValueError("Could not find reference text. Pass --ref-text or provide a matching audition manifest.")

    engine = create_tts_engine("omnivoice", lazy=False)
    if not isinstance(engine, OmniVoiceTTS):
        raise TypeError(f"Expected OmniVoiceTTS, got {type(engine).__name__}")

    voice_dir = engine.freeze_voice(
        voice_id=args.voice_id,
        ref_audio=sample_path,
        ref_text=ref_text,
        source_voice=str(row["voice"]) if row and row.get("voice") else None,
        source_sample=sample_path,
        overwrite=args.overwrite,
    )

    console.print(f"[green]Saved OmniVoice voice:[/green] {voice_dir}")
    console.print(f"[bold]Voice id:[/bold] {args.voice_id}")
    console.print("[bold]Use it with:[/bold]")
    console.print(f"  uv run python scripts/smoke_test_tts.py --model omnivoice --voice {args.voice_id} --play")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
