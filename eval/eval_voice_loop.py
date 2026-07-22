from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from rich.console import Console
from rich.progress import track
from rich.table import Table

from eval.result_io import EvalRunPaths, make_eval_run_paths, update_latest_eval_report
from eval.system_metrics import get_current_memory_mb
from soca.core import (
    DEFAULT_VOICE_RUNTIME_PROFILE_KEY,
    VOICE_RUNTIME_PROFILES,
    AudioSink,
    NullAudioPlayer,
    ResolvedVoiceRuntimeConfig,
    build_voice_runtime,
    resolve_voice_runtime_config,
)
from soca.tts import VALTEC_TTS_CONFIG, TTSRuntimeUnavailableError, create_tts_engine

console = Console(width=180)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_PATH = REPO_ROOT / "eval" / "prompts" / "voice_loop_smoke_vi.jsonl"
DEFAULT_AUDIO_DIR = REPO_ROOT / "eval" / "audio" / "voice_loop_smoke"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "eval" / "results"
ASR_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class VoiceLoopPrompt:
    prompt_id: str
    category: str
    text: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class VoiceLoopSample:
    sample_id: str
    audio_path: Path
    expected_text: str | None = None
    category: str | None = None
    tags: tuple[str, ...] = ()


def safe_filename(value: str) -> str:
    chars = [char if char.isalnum() or char in ("-", "_") else "_" for char in value]
    return "".join(chars).strip("_") or "sample"


def load_prompts(path: Path, limit: int | None = None) -> list[VoiceLoopPrompt]:
    prompts: list[VoiceLoopPrompt] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            try:
                tags = payload.get("tags", [])
                if isinstance(tags, str):
                    tags = [tags]
                prompts.append(
                    VoiceLoopPrompt(
                        prompt_id=safe_filename(str(payload["id"])),
                        category=str(payload["category"]),
                        text=str(payload["text"]),
                        tags=tuple(str(tag) for tag in tags),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{line_no} missing field: {exc}") from exc
            if limit is not None and len(prompts) >= limit:
                break
    if not prompts:
        raise ValueError(f"No voice-loop prompts loaded from {path}")
    return prompts


def collect_samples(
    *,
    audio_dir: Path | None,
    audio_files: Sequence[Path],
    prompts: Sequence[VoiceLoopPrompt] | None,
    limit: int | None = None,
) -> list[VoiceLoopSample]:
    samples: list[VoiceLoopSample] = []

    for audio_file in audio_files:
        samples.append(
            VoiceLoopSample(
                sample_id=safe_filename(audio_file.stem),
                audio_path=audio_file.expanduser().resolve(),
            )
        )

    if audio_dir is not None:
        audio_dir = audio_dir.expanduser().resolve()
        if prompts:
            for prompt in prompts:
                samples.append(
                    VoiceLoopSample(
                        sample_id=prompt.prompt_id,
                        audio_path=audio_dir / f"{prompt.prompt_id}.wav",
                        expected_text=prompt.text,
                        category=prompt.category,
                        tags=prompt.tags,
                    )
                )
        elif audio_dir.exists():
            for audio_path in sorted(audio_dir.glob("*.wav")):
                samples.append(
                    VoiceLoopSample(
                        sample_id=safe_filename(audio_path.stem),
                        audio_path=audio_path,
                    )
                )

    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError("No voice-loop audio samples selected")
    return samples


def missing_sample_paths(samples: Sequence[VoiceLoopSample]) -> list[Path]:
    return [sample.audio_path for sample in samples if not sample.audio_path.exists()]


def load_audio(path: Path, sample_rate: int = ASR_SAMPLE_RATE) -> np.ndarray:
    audio, _sr = librosa.load(str(path), sr=sample_rate, mono=True)
    return np.ascontiguousarray(audio, dtype=np.float32)


def generate_audio_fixtures(
    prompts: Sequence[VoiceLoopPrompt],
    *,
    audio_dir: Path,
    voice: str | None = None,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    audio_dir.mkdir(parents=True, exist_ok=True)
    engine = create_tts_engine(voice=voice)
    rows: list[dict[str, Any]] = []
    for prompt in prompts:
        audio_path = audio_dir / f"{prompt.prompt_id}.wav"
        if audio_path.exists() and not overwrite:
            rows.append(
                {
                    "id": prompt.prompt_id,
                    "text": prompt.text,
                    "audio_path": str(audio_path),
                    "status": "exists",
                }
            )
            continue
        result = engine.synthesize(prompt.text, voice=voice)
        sf.write(audio_path, result.audio, result.sample_rate)
        rows.append(
            {
                "id": prompt.prompt_id,
                "text": prompt.text,
                "audio_path": str(audio_path),
                "status": "generated",
                "tts_model": VALTEC_TTS_CONFIG.key,
                "voice": result.voice,
                "sample_rate": result.sample_rate,
                "audio_duration_ms": result.audio_duration_ms,
                "latency_ms": result.latency_ms,
            }
        )
    return rows


def summarize(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "min": None, "max": None}
    sorted_values = sorted(values)
    p95_index = min(int(len(sorted_values) * 0.95), len(sorted_values) - 1)
    return {
        "mean": mean(values),
        "median": median(values),
        "p95": sorted_values[p95_index],
        "min": min(values),
        "max": max(values),
    }


def cleanup_runtime() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "empty_cache"):
        try:
            mps.empty_cache()
        except RuntimeError:
            pass


def sample_audio_duration_s(audio: np.ndarray, sample_rate: int = ASR_SAMPLE_RATE) -> float:
    return len(audio) / sample_rate if sample_rate > 0 else 0.0


def evaluate_sample(
    bundle: Any,
    sample: VoiceLoopSample,
    *,
    audio_sink: AudioSink,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        audio = load_audio(sample.audio_path)
    except Exception as exc:
        return {
            "id": sample.sample_id,
            "category": sample.category,
            "tags": list(sample.tags),
            "audio_path": str(sample.audio_path),
            "expected_text": sample.expected_text,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }

    row: dict[str, Any] = {
        "id": sample.sample_id,
        "category": sample.category,
        "tags": list(sample.tags),
        "audio_path": str(sample.audio_path),
        "expected_text": sample.expected_text,
        "audio_duration_s": sample_audio_duration_s(audio),
        "status": "ok",
        "transcript": "",
        "response_text": "",
        "runtime_route": "",
        "runtime_blocked": False,
        "rejected": False,
        "repair_kind": "",
        "tts_chunks": 0,
        "tts_ready_ttfa_ms": None,
        "audible_ttfa_ms": None,
        "synthesis_slack_ms": [],
        "crossfade_fallback_count": 0,
        "output_underflow_count": 0,
        "peak_abs": 0.0,
        "total_latency_ms": None,
        "stage_latencies_ms": {},
        "errors": [],
    }

    try:
        for event in bundle.pipeline.turn_streaming(audio, audio_sink=audio_sink):
            metadata = event.metadata or {}
            if event.type == "asr":
                row["transcript"] = event.text
                row["asr_rejection_reason"] = metadata.get(
                    "rejection_reason",
                    "",
                )
            elif event.type == "runtime":
                row["response_text"] = event.text
                row["runtime_route"] = metadata.get("route", "")
                row["runtime_blocked"] = bool(metadata.get("blocked", False))
                row["used_tool"] = bool(metadata.get("used_tool", False))
                row["used_llm"] = bool(metadata.get("used_llm", False))
                row["citations"] = metadata.get("citations", [])
            elif event.type == "tts":
                row["tts_chunks"] += 1
                if row["tts_ready_ttfa_ms"] is None:
                    row["tts_ready_ttfa_ms"] = metadata.get(
                        "tts_ready_ttfa_ms",
                        metadata.get("ttfa_ms"),
                    )
                if event.tts is not None:
                    row.setdefault("tts_audio_duration_ms", 0.0)
                    row["tts_audio_duration_ms"] += event.tts.audio_duration_ms
            elif event.type == "audio":
                if row["audible_ttfa_ms"] is None:
                    row["audible_ttfa_ms"] = metadata.get("audible_ttfa_ms")
                if metadata.get("synthesis_slack_ms") is not None:
                    row["synthesis_slack_ms"].append(float(metadata["synthesis_slack_ms"]))
                row["crossfade_fallback_count"] += int(
                    metadata.get("crossfade_fallback") == "non_overlapping"
                )
                row["output_underflow_count"] = max(
                    row["output_underflow_count"],
                    int(metadata.get("output_underflow_count", 0)),
                )
                row["peak_abs"] = max(
                    row["peak_abs"],
                    float(metadata.get("peak_abs", 0.0)),
                )
            elif event.type == "error":
                row["errors"].append(event.text)
            elif event.type == "done":
                row["response_text"] = row["response_text"] or event.text
                row["rejected"] = bool(metadata.get("rejected", False))
                row["repair_kind"] = metadata.get(
                    "repair_kind",
                    row["repair_kind"],
                )
                row["runtime_blocked"] = bool(
                    metadata.get("runtime_blocked", row["runtime_blocked"])
                )
                row["runtime_route"] = metadata.get(
                    "runtime_route",
                    row["runtime_route"],
                )
                row["stage_latencies_ms"] = metadata.get(
                    "stage_latencies_ms",
                    {},
                )
                playback = metadata.get("playback", {})
                row["output_underflow_count"] = max(
                    row["output_underflow_count"],
                    int(playback.get("output_underflow_count", 0)),
                )
                row["total_latency_ms"] = event.latency_ms
    except Exception as exc:
        row["status"] = "error"
        row["errors"].append(f"{type(exc).__name__}: {exc}")
        row["total_latency_ms"] = (time.perf_counter() - started) * 1000.0

    row["ttfa_ms"] = row["audible_ttfa_ms"] or row["tts_ready_ttfa_ms"]
    if row["errors"] and row["status"] == "ok":
        row["status"] = "partial"
    return row


def runtime_config_to_dict(config: ResolvedVoiceRuntimeConfig) -> dict[str, Any]:
    return {
        "profile": config.profile_key,
        "asr_model": config.asr_model,
        "llm_model": config.llm_model,
        "tts_model": VALTEC_TTS_CONFIG.key,
        "tts_voice": config.tts_voice,
        "endpoint_silence_ms": config.endpoint_silence_ms,
        "max_record_ms": config.max_record_ms,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "vault": str(config.vault),
        "no_memory": config.no_memory,
    }


def run_profile_eval(
    profile_key: str,
    samples: Sequence[VoiceLoopSample],
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = resolve_voice_runtime_config(
        profile_key=profile_key,
        asr_model=args.asr_model,
        llm_model=args.llm_model,
        tts_voice=args.voice,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        vault=args.vault,
        no_memory=args.no_memory,
        memory_chars=args.memory_chars,
        profile_chars=args.profile_chars,
        session_chars=args.session_chars,
        session_turns=args.session_turns,
        turn_chars=args.turn_chars,
    )
    load_started = time.perf_counter()
    try:
        bundle = build_voice_runtime(config)
    except (FileNotFoundError, TTSRuntimeUnavailableError, ImportError, RuntimeError) as exc:
        return {
            "profile": profile_key,
            "config": runtime_config_to_dict(config),
            "status": "skipped_unavailable",
            "skip_reason": str(exc),
        }
    load_ms = (time.perf_counter() - load_started) * 1000

    peak_memory_mb = get_current_memory_mb()
    audio_sink: AudioSink = NullAudioPlayer()
    try:
        rows = [
            evaluate_sample(bundle, sample, audio_sink=audio_sink)
            for sample in track(
                samples,
                description=f"Benchmarking voice loop {profile_key}...",
            )
        ]
    finally:
        audio_sink.stop()
    current_memory_mb = get_current_memory_mb()
    if current_memory_mb is not None:
        peak_memory_mb = max(peak_memory_mb or current_memory_mb, current_memory_mb)

    ok_rows = [row for row in rows if row["status"] in {"ok", "partial"}]
    total_latencies = [
        float(row["total_latency_ms"]) for row in ok_rows if row.get("total_latency_ms") is not None
    ]
    ttfa_values = [float(row["ttfa_ms"]) for row in ok_rows if row.get("ttfa_ms") is not None]
    tts_ready_ttfa_values = [
        float(row["tts_ready_ttfa_ms"])
        for row in ok_rows
        if row.get("tts_ready_ttfa_ms") is not None
    ]
    audible_ttfa_values = [
        float(row["audible_ttfa_ms"]) for row in ok_rows if row.get("audible_ttfa_ms") is not None
    ]
    synthesis_slack_values = [
        float(value) for row in ok_rows for value in row.get("synthesis_slack_ms", [])
    ]
    asr_values = [
        float(row.get("stage_latencies_ms", {}).get("asr"))
        for row in ok_rows
        if row.get("stage_latencies_ms", {}).get("asr") is not None
    ]
    runtime_values = [
        float(row.get("stage_latencies_ms", {}).get("runtime"))
        for row in ok_rows
        if row.get("stage_latencies_ms", {}).get("runtime") is not None
    ]
    first_tts_values = [
        float(row.get("stage_latencies_ms", {}).get("tts_0"))
        for row in ok_rows
        if row.get("stage_latencies_ms", {}).get("tts_0") is not None
    ]
    route_counts: dict[str, int] = {}
    for row in ok_rows:
        route = str(row.get("runtime_route") or "none")
        route_counts[route] = route_counts.get(route, 0) + 1

    repair_kind_counts: dict[str, int] = {}
    for row in ok_rows:
        kind = str(row.get("repair_kind") or "")
        if kind:
            repair_kind_counts[kind] = repair_kind_counts.get(kind, 0) + 1
    repair_count = sum(repair_kind_counts.values())

    result = {
        "profile": profile_key,
        "config": runtime_config_to_dict(config),
        "status": "ok" if len(ok_rows) == len(rows) else "partial",
        "playback_sink": type(audio_sink).__name__,
        "load_ms": load_ms,
        "memory_status": bundle.memory_status,
        "knowledge_status": bundle.knowledge_status,
        "asr_guard_status": bundle.asr_guard_status,
        "sample_count": len(rows),
        "ok_count": len(ok_rows),
        "error_rate": (len(rows) - len(ok_rows)) / len(rows) if rows else 0.0,
        "reject_rate": (
            sum(bool(row.get("rejected")) for row in ok_rows) / len(ok_rows) if ok_rows else 0.0
        ),
        "runtime_block_rate": (
            sum(bool(row.get("runtime_blocked")) for row in ok_rows) / len(ok_rows)
            if ok_rows
            else 0.0
        ),
        "avg_tts_chunks": mean([int(row.get("tts_chunks", 0)) for row in ok_rows])
        if ok_rows
        else 0.0,
        "route_counts": dict(sorted(route_counts.items())),
        "repair_count": repair_count,
        "repair_kind_counts": dict(sorted(repair_kind_counts.items())),
        "total_latency_ms": summarize(total_latencies),
        "ttfa_ms": summarize(ttfa_values),
        "tts_ready_ttfa_ms": summarize(tts_ready_ttfa_values),
        "audible_ttfa_ms": summarize(audible_ttfa_values),
        "synthesis_slack_ms": summarize(synthesis_slack_values),
        "crossfade_fallback_count": sum(
            int(row.get("crossfade_fallback_count", 0)) for row in ok_rows
        ),
        "output_underflow_count": sum(int(row.get("output_underflow_count", 0)) for row in ok_rows),
        "peak_abs": max((float(row.get("peak_abs", 0.0)) for row in ok_rows), default=0.0),
        "asr_latency_ms": summarize(asr_values),
        "runtime_latency_ms": summarize(runtime_values),
        "first_tts_latency_ms": summarize(first_tts_values),
        "peak_memory_mb": peak_memory_mb,
        "rows": rows,
    }
    del bundle
    cleanup_runtime()
    return result


def format_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}"


def format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def render_summary(results: Sequence[dict[str, Any]]) -> None:
    table = Table(title="SoCa E2E Voice Loop Benchmark", show_lines=True)
    table.add_column("Profile", style="cyan")
    table.add_column("ASR")
    table.add_column("LLM")
    table.add_column("TTS")
    table.add_column("Voice")
    table.add_column("Status")
    table.add_column("Load ms", justify="right")
    table.add_column("TTFA p50", justify="right")
    table.add_column("TTFA p95", justify="right")
    table.add_column("Total p50", justify="right")
    table.add_column("Total p95", justify="right")
    table.add_column("Reject", justify="right")
    table.add_column("Err", justify="right")
    table.add_column("Peak MB", justify="right")
    table.add_column("Skip reason", overflow="fold", width=44)

    for result in results:
        config = result.get("config", {})
        if result["status"] == "skipped_unavailable":
            table.add_row(
                result["profile"],
                config.get("asr_model", ""),
                config.get("llm_model", ""),
                config.get("tts_model", ""),
                str(config.get("tts_voice", "")),
                "skipped",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                "n/a",
                result.get("skip_reason", ""),
            )
            continue

        ttfa = result.get("ttfa_ms", {})
        total = result.get("total_latency_ms", {})
        peak_memory = result.get("peak_memory_mb")
        table.add_row(
            result["profile"],
            config.get("asr_model", ""),
            config.get("llm_model", ""),
            config.get("tts_model", ""),
            str(config.get("tts_voice", "")),
            result["status"],
            f"{result.get('load_ms', 0):.0f}",
            format_ms(ttfa.get("median")),
            format_ms(ttfa.get("p95")),
            format_ms(total.get("median")),
            format_ms(total.get("p95")),
            format_percent(result.get("reject_rate")),
            format_percent(result.get("error_rate")),
            "n/a" if peak_memory is None else f"{peak_memory:.0f}",
            result.get("skip_reason", ""),
        )
    console.print(table)


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_outputs(
    results: Sequence[dict[str, Any]],
    run_paths: EvalRunPaths,
    *,
    samples: Sequence[VoiceLoopSample],
    fixture_generation: Sequence[dict[str, Any]],
) -> tuple[Path, Path]:
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": run_paths.run_dir.name,
        "samples": [
            {
                "id": sample.sample_id,
                "audio_path": str(sample.audio_path),
                "expected_text": sample.expected_text,
                "category": sample.category,
                "tags": list(sample.tags),
            }
            for sample in samples
        ],
        "fixture_generation": list(fixture_generation),
        "results": list(results),
    }
    run_paths.json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# SoCa E2E Voice Loop Benchmark",
        "",
        f"- Created at: `{run_paths.run_dir.name}`",
        f"- Sample count: `{len(samples)}`",
        f"- Fixture generation rows: `{len(fixture_generation)}`",
        "- Playback: `NullAudioPlayer` (audio is synthesized but not sent to speakers)",
        "",
        "| Profile | ASR | LLM | TTS | Voice | Status | Load ms | TTFA p50 | TTFA p95 | Total p50 | Total p95 | Reject | Err | Peak MB | Skip reason |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        config = result.get("config", {})
        if result["status"] == "skipped_unavailable":
            lines.append(
                f"| {result['profile']} | {config.get('asr_model', '')} | "
                f"{config.get('llm_model', '')} | {config.get('tts_model', '')} | "
                f"{config.get('tts_voice', '')} | skipped | n/a | n/a | n/a | n/a | n/a | "
                f"n/a | n/a | n/a | {markdown_cell(result.get('skip_reason', ''))} |"
            )
            continue
        ttfa = result.get("ttfa_ms", {})
        total = result.get("total_latency_ms", {})
        peak_memory = result.get("peak_memory_mb")
        lines.append(
            f"| {result['profile']} | {config.get('asr_model', '')} | "
            f"{config.get('llm_model', '')} | {config.get('tts_model', '')} | "
            f"{config.get('tts_voice', '')} | {result['status']} | "
            f"{result.get('load_ms', 0):.0f} | {format_ms(ttfa.get('median'))} | "
            f"{format_ms(ttfa.get('p95'))} | {format_ms(total.get('median'))} | "
            f"{format_ms(total.get('p95'))} | {format_percent(result.get('reject_rate'))} | "
            f"{format_percent(result.get('error_rate'))} | "
            f"{'n/a' if peak_memory is None else f'{peak_memory:.0f}'} | "
            f"{markdown_cell(result.get('skip_reason', ''))} |"
        )

    lines.extend(["", "## Samples", ""])
    lines.append("| ID | Category | Expected text | Audio path |")
    lines.append("|---|---|---|---|")
    for sample in samples:
        lines.append(
            f"| {sample.sample_id} | {sample.category or ''} | "
            f"{markdown_cell(sample.expected_text or '')} | `{sample.audio_path}` |"
        )

    lines.extend(["", "## Per-Sample Results", ""])
    lines.append(
        "| Profile | ID | Transcript | Route | ASR ms | Runtime ms | TTS0 ms | TTFA ms | Total ms | Chunks | Status |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    for result in results:
        for row in result.get("rows", []):
            stages = row.get("stage_latencies_ms", {})
            lines.append(
                f"| {result['profile']} | {row.get('id', '')} | "
                f"{markdown_cell(row.get('transcript', ''))} | "
                f"{markdown_cell(row.get('runtime_route', ''))} | "
                f"{format_ms(stages.get('asr'))} | {format_ms(stages.get('runtime'))} | "
                f"{format_ms(stages.get('tts_0'))} | {format_ms(row.get('ttfa_ms'))} | "
                f"{format_ms(row.get('total_latency_ms'))} | {row.get('tts_chunks', 0)} | "
                f"{row.get('status', '')} |"
            )

    run_paths.md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    update_latest_eval_report(run_paths)
    return run_paths.json_path, run_paths.md_path


def parse_profiles(values: Sequence[str], all_profiles: bool) -> list[str]:
    if all_profiles:
        return list(VOICE_RUNTIME_PROFILES)
    profiles: list[str] = []
    for value in values:
        profiles.extend(part.strip() for part in value.split(",") if part.strip())
    if not profiles:
        profiles = [DEFAULT_VOICE_RUNTIME_PROFILE_KEY]
    unknown = sorted(set(profiles) - set(VOICE_RUNTIME_PROFILES))
    if unknown:
        valid = ", ".join(sorted(VOICE_RUNTIME_PROFILES))
        raise ValueError(f"Unknown profile(s): {', '.join(unknown)}. Valid profiles: {valid}")
    deduped: list[str] = []
    seen: set[str] = set()
    for profile in profiles:
        if profile not in seen:
            deduped.append(profile)
            seen.add(profile)
    return deduped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SoCa E2E voice-loop benchmark.")
    parser.add_argument(
        "--profile", action="append", default=[], help="Runtime profile. Can be comma-separated."
    )
    parser.add_argument("--all-profiles", action="store_true", help="Run every runtime profile.")
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--audio-file", action="append", default=[], type=Path)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vault", type=Path, default=Path.home() / "KnowledgeVault")
    parser.add_argument("--no-memory", action="store_true")
    parser.add_argument("--memory-chars", type=int, default=2200)
    parser.add_argument("--profile-chars", type=int, default=900)
    parser.add_argument("--session-chars", type=int, default=1300)
    parser.add_argument("--session-turns", type=int, default=6)
    parser.add_argument("--turn-chars", type=int, default=500)
    parser.add_argument("--asr-model", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--voice", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--no-playback",
        action="store_true",
        help="Accepted for clarity; benchmark always uses NullAudioPlayer.",
    )
    parser.add_argument(
        "--generate-fixtures",
        action="store_true",
        help="Generate missing fixture WAV files from --prompts before benchmarking.",
    )
    parser.add_argument("--overwrite-fixtures", action="store_true")
    parser.add_argument("--fixture-voice", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profiles = parse_profiles(args.profile, args.all_profiles)
    prompts = load_prompts(args.prompts, limit=args.limit) if args.prompts else None

    if args.audio_file and args.generate_fixtures:
        raise ValueError("--audio-file and --generate-fixtures cannot be used together.")

    fixture_generation: list[dict[str, Any]] = []
    if args.generate_fixtures:
        if not prompts:
            raise ValueError("--generate-fixtures requires --prompts")
        fixture_generation = generate_audio_fixtures(
            prompts,
            audio_dir=args.audio_dir,
            voice=args.fixture_voice,
            overwrite=args.overwrite_fixtures,
        )

    sample_prompts = None if args.audio_file else prompts
    sample_audio_dir = None if args.audio_file else args.audio_dir
    samples = collect_samples(
        audio_dir=sample_audio_dir,
        audio_files=args.audio_file,
        prompts=sample_prompts,
        limit=args.limit,
    )
    missing = missing_sample_paths(samples)
    if missing:
        missing_preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(
            "Missing voice-loop fixture audio files:\n"
            f"{missing_preview}\n"
            "Run again with --generate-fixtures, or pass --audio-file for existing WAVs."
        )

    console.print(
        f"[bold]E2E voice-loop benchmark[/bold]: {len(profiles)} profile(s), "
        f"{len(samples)} sample(s), playback=NullAudioPlayer"
    )
    created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_paths = make_eval_run_paths(args.output_dir, "voice_loop", created_at)

    results = [run_profile_eval(profile, samples, args=args) for profile in profiles]
    render_summary(results)
    json_path, md_path = write_outputs(
        results,
        run_paths,
        samples=samples,
        fixture_generation=fixture_generation,
    )
    console.print(f"\n[green]Saved[/green] {json_path}")
    console.print(f"[green]Saved[/green] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
