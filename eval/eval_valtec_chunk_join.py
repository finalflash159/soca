"""Build fair hard-join/cross-fade Valtec clips from identical synthesized chunks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import soundfile as sf

from soca.core.audio_join import crossfade_pcm
from soca.core.text_chunking import chunk_text_for_tts, split_first_clause
from soca.tts import VALTEC_TTS_CONFIG, create_tts_engine


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    text: str
    category: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prompts(path: Path) -> list[Prompt]:
    prompts: list[Prompt] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            prompt_id = str(payload.get("id", "")).strip()
            text = str(payload.get("text", "")).strip()
            category = str(payload.get("category", "")).strip()
            if not prompt_id or not text or not category:
                raise ValueError(
                    f"{path}:{line_number} requires non-empty id, text and category"
                )
            prompts.append(
                Prompt(prompt_id=prompt_id, text=text, category=category)
            )
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def chunk_like_runtime(text: str) -> list[str]:
    first, remainder = split_first_clause(
        text,
        min_chars=12,
        min_words=2,
        max_scan_chars=80,
    )
    if first is None:
        return chunk_text_for_tts(text, min_chars=24)
    return [first, *chunk_text_for_tts(remainder, min_chars=24)]


def join_chunks(
    chunks: Sequence[np.ndarray],
    *,
    sample_rate: int,
    fade_ms: float,
) -> np.ndarray:
    if not chunks:
        return np.empty(0, dtype=np.float32)
    joined = np.asarray(chunks[0], dtype=np.float32).reshape(-1)
    for chunk in chunks[1:]:
        joined = crossfade_pcm(
            joined,
            chunk,
            sample_rate=sample_rate,
            fade_ms=fade_ms,
        )
    return np.ascontiguousarray(joined, dtype=np.float32)


def boundary_jumps(chunks: Sequence[np.ndarray]) -> list[float]:
    return [
        abs(float(left[-1]) - float(right[0]))
        for left, right in zip(chunks, chunks[1:], strict=False)
        if left.size and right.size
    ]


def joined_boundary_region_jumps(
    chunks: Sequence[np.ndarray],
    *,
    sample_rate: int,
    fade_ms: float,
) -> list[float]:
    """Đo bước nhảy lớn nhất quanh từng vùng nối sau khi đã join."""
    if len(chunks) < 2:
        return []
    joined = np.asarray(chunks[0], dtype=np.float32).reshape(-1)
    jumps: list[float] = []
    requested = max(0, int(round(sample_rate * fade_ms / 1000.0)))
    for right_raw in chunks[1:]:
        right = np.asarray(right_raw, dtype=np.float32).reshape(-1)
        left_length = len(joined)
        overlap = min(requested, left_length, len(right))
        joined = crossfade_pcm(
            joined,
            right,
            sample_rate=sample_rate,
            fade_ms=fade_ms,
        )
        region_start = max(0, left_length - overlap - 1)
        region_stop = min(len(joined), left_length + 1)
        region = joined[region_start:region_stop]
        jumps.append(
            float(np.max(np.abs(np.diff(region))))
            if region.size >= 2
            else 0.0
        )
    return jumps


def summarize(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(median(values)),
        "p95": float(np.percentile(array, 95)),
    }


def safe_name(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in value
    ).strip("_") or "sample"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", choices=("current",), default="current")
    parser.add_argument("--voices", default="NF,SF,NM1,SM,NM2")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    del args.artifact
    prompts = load_prompts(args.prompts)
    voices = [voice.strip() for voice in args.voices.split(",") if voice.strip()]
    if not voices:
        raise ValueError("--voices must select at least one voice")

    engine = create_tts_engine(voice=voices[0])
    available = set(engine.list_voices())
    unknown = sorted(set(voices) - available)
    if unknown:
        raise ValueError(f"Unknown Valtec voices: {unknown}; available: {sorted(available)}")
    artifact_metadata = engine.artifact_metadata

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    all_chunk_latencies: list[float] = []
    all_hard_jumps: list[float] = []
    for voice in voices:
        for prompt in prompts:
            chunk_texts = chunk_like_runtime(prompt.text)
            results = [engine.synthesize(text, voice=voice) for text in chunk_texts]
            sample_rates = {result.sample_rate for result in results}
            if len(sample_rates) != 1:
                raise ValueError(
                    f"Sample-rate drift for {prompt.prompt_id}/{voice}: {sample_rates}"
                )
            sample_rate = sample_rates.pop()
            audio_chunks = [
                np.asarray(result.audio, dtype=np.float32).reshape(-1)
                for result in results
            ]
            hard = join_chunks(audio_chunks, sample_rate=sample_rate, fade_ms=0.0)
            fade_8 = join_chunks(audio_chunks, sample_rate=sample_rate, fade_ms=8.0)
            fade_12 = join_chunks(audio_chunks, sample_rate=sample_rate, fade_ms=12.0)
            variants = {"hard": hard, "equal_gain_8ms": fade_8, "equal_gain_12ms": fade_12}

            outputs: dict[str, dict[str, Any]] = {}
            fade_by_variant = {
                "hard": 0.0,
                "equal_gain_8ms": 8.0,
                "equal_gain_12ms": 12.0,
            }
            for variant, audio in variants.items():
                path = args.output_dir / (
                    f"{safe_name(prompt.prompt_id)}_{safe_name(voice)}_{variant}.wav"
                )
                sf.write(path, audio, sample_rate, subtype="FLOAT")
                outputs[variant] = {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "samples": len(audio),
                    "peak_abs": float(np.max(np.abs(audio))) if audio.size else 0.0,
                    "boundary_region_max_adjacent_jumps": joined_boundary_region_jumps(
                        audio_chunks,
                        sample_rate=sample_rate,
                        fade_ms=fade_by_variant[variant],
                    ),
                }

            jumps = boundary_jumps(audio_chunks)
            all_hard_jumps.extend(jumps)
            all_chunk_latencies.extend(result.latency_ms for result in results)
            rows.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "text": prompt.text,
                    "category": prompt.category,
                    "voice": voice,
                    "artifact_manifest_sha256": artifact_metadata["manifest_sha256"],
                    "chunks": [
                        {
                            "text": text,
                            "latency_ms": result.latency_ms,
                            "duration_ms": result.audio_duration_ms,
                            "rtf": result.rtf,
                        }
                        for text, result in zip(chunk_texts, results, strict=True)
                    ],
                    "boundary_jumps_before_join": jumps,
                    "outputs": outputs,
                }
            )

    report = {
        "schema_version": 1,
        "model": VALTEC_TTS_CONFIG.key,
        "artifact": artifact_metadata,
        "prompt_file": str(args.prompts.resolve()),
        "prompt_sha256": _sha256(args.prompts),
        "voices": voices,
        "chunk_latency_ms": summarize(all_chunk_latencies),
        "hard_boundary_jump": summarize(all_hard_jumps),
        "rows": rows,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    listening_path = args.output_dir / "listening.csv"
    with listening_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "prompt_id",
            "category",
            "voice",
            "variant",
            "wav_path",
            "wav_sha256",
            "reviewer",
            "device",
            "click",
            "gap",
            "double_speech",
            "prosody_score_1_to_5",
            "pronunciation_score_1_to_5",
            "preferred_variant",
            "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for variant, output in row["outputs"].items():
                writer.writerow(
                    {
                        "prompt_id": row["prompt_id"],
                        "category": row["category"],
                        "voice": row["voice"],
                        "variant": variant,
                        "wav_path": output["path"],
                        "wav_sha256": output["sha256"],
                        "reviewer": "",
                        "device": "",
                        "click": "",
                        "gap": "",
                        "double_speech": "",
                        "prosody_score_1_to_5": "",
                        "pronunciation_score_1_to_5": "",
                        "preferred_variant": "",
                        "notes": "",
                    }
                )
    print(report_path)
    print(listening_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
