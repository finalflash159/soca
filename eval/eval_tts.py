"""Benchmark local Vietnamese TTS engines on a fixed prompt set."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import soundfile as sf
from rich.console import Console
from rich.progress import track
from rich.table import Table

from eval.result_io import EvalRunPaths, make_eval_run_paths, update_latest_eval_report
from eval.system_metrics import get_current_memory_mb
from soca.core.profiles import VOICE_RUNTIME_PROFILES, get_voice_runtime_profile
from soca.tts import (
    TIER_A_TTS_MODEL_KEYS,
    TIER_B_TTS_MODEL_KEYS,
    TTS_MODEL_REGISTRY,
    TTSRuntimeUnavailableError,
    create_tts_engine,
)
from soca.tts.registry import DEFAULT_TTS_MODEL_KEY, get_tts_model_config

console = Console(width=180)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_PATH = REPO_ROOT / "eval" / "prompts" / "tts_bakeoff_vi.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "eval" / "results"
VOICE_POLICIES = ("default", "smoke", "all")
REQUIRED_PROMPT_CATEGORIES = (
    "short",
    "assistant",
    "coach",
    "nutrition",
    "fitness",
    "safety",
    "tracking",
    "number",
    "datetime",
    "currency",
    "measurement",
    "name_place",
    "abbreviation",
    "punctuation",
    "codeswitch",
    "formal",
    "casual",
    "asr_noisy",
    "long",
)


@dataclass(frozen=True)
class TTSPrompt:
    prompt_id: str
    category: str
    text: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class TTSModelEvalTarget:
    model_key: str
    profile_key: str | None = None
    requested_voice: str | None = None
    voice_source: str | None = None


def safe_filename(value: str) -> str:
    chars = [char if char.isalnum() or char in ("-", "_") else "_" for char in value]
    return "".join(chars).strip("_") or "tts"


def load_prompts(path: Path, limit: int | None = None) -> list[TTSPrompt]:
    prompts: list[TTSPrompt] = []
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
                    TTSPrompt(
                        prompt_id=str(payload["id"]),
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
        raise ValueError(f"No TTS prompts loaded from {path}")
    return prompts


def summarize_prompt_coverage(prompts: Sequence[TTSPrompt]) -> dict[str, Any]:
    categories: dict[str, int] = {}
    tags: dict[str, int] = {}
    for prompt in prompts:
        categories[prompt.category] = categories.get(prompt.category, 0) + 1
        for tag in prompt.tags:
            tags[tag] = tags.get(tag, 0) + 1

    missing_required_categories = [
        category for category in REQUIRED_PROMPT_CATEGORIES if categories.get(category, 0) == 0
    ]
    return {
        "total_prompts": len(prompts),
        "categories": dict(sorted(categories.items())),
        "tags": dict(sorted(tags.items())),
        "required_categories": list(REQUIRED_PROMPT_CATEGORIES),
        "missing_required_categories": missing_required_categories,
    }


def validate_prompt_coverage(prompts: Sequence[TTSPrompt]) -> None:
    coverage = summarize_prompt_coverage(prompts)
    missing = coverage["missing_required_categories"]
    if missing:
        raise ValueError(
            "Prompt corpus is missing required TTS coverage categories: " + ", ".join(missing)
        )


def dedupe_model_keys(model_keys: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for model_key in model_keys:
        if model_key in seen:
            continue
        seen.add(model_key)
        result.append(model_key)
    return result


def parse_model_list(values: Sequence[str]) -> list[str]:
    model_keys: list[str] = []
    for value in values:
        model_keys.extend(part.strip() for part in value.split(",") if part.strip())

    unknown = sorted(set(model_keys) - set(TTS_MODEL_REGISTRY))
    if unknown:
        valid = ", ".join(sorted(TTS_MODEL_REGISTRY))
        raise ValueError(f"Unknown TTS model key(s): {', '.join(unknown)}. Valid keys: {valid}")
    return model_keys


def parse_voice_map(values: Sequence[str]) -> dict[str, str]:
    voice_map: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --voice-map {value!r}; expected MODEL=VOICE.")
        model_key, voice = value.split("=", 1)
        model_key = model_key.strip()
        voice = voice.strip()
        if model_key not in TTS_MODEL_REGISTRY:
            valid = ", ".join(sorted(TTS_MODEL_REGISTRY))
            raise ValueError(f"Unknown TTS model key in --voice-map: {model_key}. Valid keys: {valid}")
        if not voice:
            raise ValueError(f"Voice for --voice-map {model_key}=... must not be empty.")
        voice_map[model_key] = voice
    return voice_map


def select_model_keys(
    model_keys: Sequence[str],
    tier_a: bool,
    tier_b: bool = False,
    all_models: bool = False,
) -> list[str]:
    selected: list[str] = []
    if all_models:
        selected.extend(TTS_MODEL_REGISTRY)
    if tier_a:
        selected.extend(TIER_A_TTS_MODEL_KEYS)
    if tier_b:
        selected.extend(TIER_B_TTS_MODEL_KEYS)
    selected.extend(model_keys)
    if not selected:
        selected.append(DEFAULT_TTS_MODEL_KEY)
    return dedupe_model_keys(selected)


def tts_tier(model_key: str) -> str:
    if model_key in TIER_A_TTS_MODEL_KEYS:
        return "A"
    if model_key in TIER_B_TTS_MODEL_KEYS:
        return "B"
    return "custom"


def build_eval_targets(
    model_keys: Sequence[str],
    profile_keys: Sequence[str],
    *,
    voice: str | None,
    voice_map: dict[str, str],
) -> list[TTSModelEvalTarget]:
    target_count = len(model_keys) + len(profile_keys)
    if voice and target_count != 1:
        raise ValueError("--voice can only be used when evaluating exactly one model or one profile.")
    if voice and voice_map:
        raise ValueError("--voice and --voice-map are mutually exclusive.")

    selected_models = set(model_keys)
    targets: list[TTSModelEvalTarget] = []

    for profile_key in profile_keys:
        profile = get_voice_runtime_profile(profile_key)
        config = get_tts_model_config(profile.tts_model)
        selected_models.add(profile.tts_model)
        if voice:
            requested_voice = voice
            voice_source = "cli"
        elif profile.tts_model in voice_map:
            requested_voice = voice_map[profile.tts_model]
            voice_source = "voice_map"
        else:
            requested_voice = profile.tts_voice or config.default_voice
            voice_source = "profile" if profile.tts_voice else "registry_default"
        targets.append(
            TTSModelEvalTarget(
                model_key=profile.tts_model,
                profile_key=profile_key,
                requested_voice=requested_voice,
                voice_source=voice_source,
            )
        )

    unused_voice_map_keys = sorted(set(voice_map) - selected_models)
    if unused_voice_map_keys:
        raise ValueError(
            "--voice-map contains model(s) that are not selected for this run: "
            + ", ".join(unused_voice_map_keys)
        )

    for model_key in model_keys:
        if voice:
            requested_voice = voice
            voice_source = "cli"
        elif model_key in voice_map:
            requested_voice = voice_map[model_key]
            voice_source = "voice_map"
        else:
            requested_voice = None
            voice_source = None
        targets.append(
            TTSModelEvalTarget(
                model_key=model_key,
                requested_voice=requested_voice,
                voice_source=voice_source,
            )
        )

    return targets


def summarize(values: Sequence[float]) -> dict[str, float]:
    sorted_values = sorted(values)
    p95_index = min(int(len(sorted_values) * 0.95), len(sorted_values) - 1)
    return {
        "mean": mean(values),
        "median": median(values),
        "p95": sorted_values[p95_index],
        "min": min(values),
        "max": max(values),
    }


def clipping_ratio(audio: np.ndarray, threshold: float = 0.995) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.mean(np.abs(audio) >= threshold))


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


def resolve_eval_voices(
    *,
    target: TTSModelEvalTarget,
    available_voices: Sequence[str],
    default_voice: str,
    smoke_test_voices: Sequence[str] | None,
    voice_policy: str,
    max_voices: int | None,
) -> list[tuple[str, str]]:
    if target.requested_voice:
        return [(target.requested_voice, target.voice_source or "cli")]

    if voice_policy == "default":
        voices = [(default_voice, "registry_default")]
    elif voice_policy == "smoke":
        source = "smoke_test_voices" if smoke_test_voices else "registry_default"
        voices = [(voice, source) for voice in (smoke_test_voices or (default_voice,))]
    elif voice_policy == "all":
        voices = [(voice, "engine_list") for voice in available_voices]
        if not voices:
            voices = [(default_voice, "registry_default")]
    else:
        raise ValueError(f"Unsupported voice policy: {voice_policy}")

    if max_voices is not None:
        voices = voices[:max_voices]
    return voices


def run_model_eval(
    target: str | TTSModelEvalTarget,
    prompts: Sequence[TTSPrompt],
    *,
    voice: str | None = None,
    voice_policy: str = "default",
    max_voices: int | None = None,
    skip_unavailable: bool = True,
    audio_dir: Path | None = None,
) -> dict[str, Any] | None:
    if isinstance(target, str):
        target = TTSModelEvalTarget(model_key=target, requested_voice=voice, voice_source="cli" if voice else None)
    elif voice is not None:
        target = TTSModelEvalTarget(
            model_key=target.model_key,
            profile_key=target.profile_key,
            requested_voice=voice,
            voice_source="cli",
        )

    model_key = target.model_key
    config = get_tts_model_config(model_key)

    console.print(f"\n[bold]Loading[/bold] {model_key} ({config.runner})")
    load_start = time.perf_counter()
    try:
        engine = create_tts_engine(model_key, voice=target.requested_voice, lazy=False)
    except (NotImplementedError, TTSRuntimeUnavailableError) as exc:
        message = f"{model_key}: {exc}"
        if skip_unavailable:
            console.print(f"[yellow]Skipping[/yellow] {message}")
            return {
                "profile": target.profile_key,
                "model": model_key,
                "tier": tts_tier(model_key),
                "role": config.role,
                "runner": config.runner,
                "status": "skipped_unavailable",
                "skip_reason": str(exc),
                "license": config.license,
                "source_url": config.source_url,
            }
        raise
    load_ms = (time.perf_counter() - load_start) * 1000

    try:
        available_voices = engine.list_voices()
    except TTSRuntimeUnavailableError as exc:
        del engine
        cleanup_runtime()
        if skip_unavailable:
            console.print(f"[yellow]Skipping[/yellow] {model_key}: {exc}")
            return {
                "profile": target.profile_key,
                "model": model_key,
                "tier": tts_tier(model_key),
                "role": config.role,
                "runner": config.runner,
                "status": "skipped_unavailable",
                "skip_reason": str(exc),
                "license": config.license,
                "source_url": config.source_url,
            }
        raise
    eval_voices = resolve_eval_voices(
        target=target,
        available_voices=available_voices,
        default_voice=config.default_voice,
        smoke_test_voices=config.smoke_test_voices,
        voice_policy=voice_policy,
        max_voices=max_voices,
    )

    rows: list[dict[str, Any]] = []
    peak_memory_mb = get_current_memory_mb()
    total_jobs = len(prompts) * len(eval_voices)

    for voice_index, (selected_voice, voice_source) in enumerate(eval_voices, start=1):
        for prompt in track(
            prompts,
            description=f"Benchmarking {model_key}/{selected_voice} ({voice_index}/{len(eval_voices)})...",
            total=len(prompts),
        ):
            try:
                result = engine.synthesize(prompt.text, voice=selected_voice)
                audio = result.audio
                audio_path = None
                if audio_dir is not None:
                    audio_dir.mkdir(parents=True, exist_ok=True)
                    audio_path = (
                        audio_dir
                        / f"{safe_filename(model_key)}__{safe_filename(selected_voice)}__{safe_filename(prompt.prompt_id)}.wav"
                    )
                    sf.write(audio_path, audio, result.sample_rate)
                row = {
                    "id": prompt.prompt_id,
                    "category": prompt.category,
                    "text": prompt.text,
                    "status": "ok",
                    "sample_rate": result.sample_rate,
                    "n_samples": int(len(audio)),
                    "latency_ms": result.latency_ms,
                    "audio_duration_ms": result.audio_duration_ms,
                    "rtf": result.rtf,
                    "requested_voice": selected_voice,
                    "voice": result.voice,
                    "voice_source": voice_source,
                    "engine": result.engine,
                    "peak_abs": float(np.max(np.abs(audio))) if audio.size else 0.0,
                    "clipping_ratio": clipping_ratio(audio),
                    "audio_path": str(audio_path) if audio_path is not None else None,
                }
            except Exception as exc:
                row = {
                    "id": prompt.prompt_id,
                    "category": prompt.category,
                    "text": prompt.text,
                    "status": "error",
                    "requested_voice": selected_voice,
                    "voice_source": voice_source,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            current_memory_mb = get_current_memory_mb()
            if current_memory_mb is not None:
                peak_memory_mb = max(peak_memory_mb or current_memory_mb, current_memory_mb)

    ok_rows = [row for row in rows if row["status"] == "ok"]
    error_rows = [row for row in rows if row["status"] != "ok"]
    if not ok_rows:
        result_payload = {
            "profile": target.profile_key,
            "model": model_key,
            "tier": tts_tier(model_key),
            "role": config.role,
            "runner": config.runner,
            "status": "failed",
            "license": config.license,
            "source_url": config.source_url,
            "load_ms": load_ms,
            "voices": [
                {"voice": selected_voice, "voice_source": voice_source}
                for selected_voice, voice_source in eval_voices
            ],
            "available_voices": available_voices,
            "total_jobs": total_jobs,
            "errors": rows,
        }
        del engine
        cleanup_runtime()
        return result_payload

    latencies = [float(row["latency_ms"]) for row in ok_rows]
    rtfs = [float(row["rtf"]) for row in ok_rows]
    durations = [float(row["audio_duration_ms"]) for row in ok_rows]
    clipping = [float(row["clipping_ratio"]) for row in ok_rows]

    result_payload = {
        "profile": target.profile_key,
        "model": model_key,
        "tier": tts_tier(model_key),
        "role": config.role,
        "runner": config.runner,
        "status": "ok" if not error_rows else "partial",
        "license": config.license,
        "source_url": config.source_url,
        "load_ms": load_ms,
        "voices": [
            {"voice": selected_voice, "voice_source": voice_source}
            for selected_voice, voice_source in eval_voices
        ],
        "available_voices": available_voices,
        "voice_policy": voice_policy,
        "total_jobs": total_jobs,
        "latency_ms": summarize(latencies),
        "rtf": summarize(rtfs),
        "audio_duration_ms": summarize(durations),
        "mean_clipping_ratio": mean(clipping),
        "non_empty_rate": sum(row["n_samples"] > 0 for row in ok_rows) / len(rows),
        "error_rate": len(error_rows) / len(rows),
        "peak_memory_mb": peak_memory_mb,
        "rows": rows,
    }
    del engine
    cleanup_runtime()
    return result_payload


def render_summary(results: Sequence[dict[str, Any]]) -> None:
    table = Table(title="Soca TTS Bakeoff", show_lines=True)
    table.add_column("Profile", style="magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Tier", justify="center")
    table.add_column("Runner")
    table.add_column("Voices", overflow="fold")
    table.add_column("Status")
    table.add_column("Load ms", justify="right")
    table.add_column("Lat p50", justify="right")
    table.add_column("Lat p95", justify="right")
    table.add_column("RTF p50", justify="right")
    table.add_column("RTF p95", justify="right")
    table.add_column("Non-empty", justify="right")
    table.add_column("Err", justify="right")
    table.add_column("Peak MB", justify="right")
    table.add_column("Skip reason", overflow="fold", width=44)

    for result in results:
        if result["status"] in {"skipped_unavailable", "worker_failed"}:
            table.add_row(
                result.get("profile") or "",
                result["model"],
                result["tier"],
                result["runner"],
                "n/a",
                "skipped" if result["status"] == "skipped_unavailable" else "worker_failed",
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

        latency = result.get("latency_ms", {})
        rtf = result.get("rtf", {})
        peak_memory = result.get("peak_memory_mb")
        voices = ", ".join(
            f"{item['voice']}:{item['voice_source']}" for item in result.get("voices", [])
        )
        table.add_row(
            result.get("profile") or "",
            result["model"],
            result["tier"],
            result["runner"],
            voices,
            result["status"],
            f"{result.get('load_ms', 0):.0f}",
            f"{latency.get('median', 0):.0f}",
            f"{latency.get('p95', 0):.0f}",
            f"{rtf.get('median', 0):.2f}",
            f"{rtf.get('p95', 0):.2f}",
            f"{result.get('non_empty_rate', 0):.1%}",
            f"{result.get('error_rate', 0):.1%}",
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
    audio_dir: Path | None,
    prompt_coverage: dict[str, Any],
) -> tuple[Path, Path]:
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_paths.json_path
    md_path = run_paths.md_path

    json_path.write_text(
        json.dumps(
            {
                "created_at": run_paths.run_dir.name,
                "audio_dir": str(audio_dir) if audio_dir is not None else None,
                "prompt_coverage": prompt_coverage,
                "results": list(results),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Soca TTS Bakeoff",
        "",
        f"- Created at: `{run_paths.run_dir.name}`",
        f"- Audio dir: `{audio_dir}`" if audio_dir is not None else "- Audio dir: disabled",
        f"- Prompt count: `{prompt_coverage['total_prompts']}`",
        f"- Prompt categories: `{json.dumps(prompt_coverage['categories'], ensure_ascii=False)}`",
        f"- Missing required categories: `{prompt_coverage['missing_required_categories']}`",
        "",
        "| Profile | Model | Tier | Runner | Voices | Status | Load ms | Lat p50 | Lat p95 | RTF p50 | RTF p95 | Non-empty | Err | Peak MB | Skip reason |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        if result["status"] in {"skipped_unavailable", "worker_failed"}:
            lines.append(
                f"| {result.get('profile') or ''} | {result['model']} | {result['tier']} | "
                f"{result['runner']} | n/a | "
                f"{'skipped' if result['status'] == 'skipped_unavailable' else 'worker_failed'} | "
                f"n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
                f" {markdown_cell(result.get('skip_reason', ''))} |"
            )
            continue
        latency = result.get("latency_ms", {})
        rtf = result.get("rtf", {})
        peak_memory = result.get("peak_memory_mb")
        voices = ", ".join(
            f"{item['voice']}:{item['voice_source']}" for item in result.get("voices", [])
        )
        lines.append(
            f"| {result.get('profile') or ''} | {result['model']} | {result['tier']} | "
            f"{result['runner']} | {voices} | {result['status']} | "
            f"{result.get('load_ms', 0):.0f} | "
            f"{latency.get('median', 0):.0f} | {latency.get('p95', 0):.0f} | "
            f"{rtf.get('median', 0):.2f} | {rtf.get('p95', 0):.2f} | "
            f"{result.get('non_empty_rate', 0):.1%} | {result.get('error_rate', 0):.1%} | "
            f"{'n/a' if peak_memory is None else f'{peak_memory:.0f}'} | "
            f"{markdown_cell(result.get('skip_reason', ''))} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    update_latest_eval_report(run_paths)
    return json_path, md_path


def worker_payload(target: TTSModelEvalTarget) -> str:
    return json.dumps(asdict(target), ensure_ascii=False)


def target_from_worker_payload(payload: str) -> TTSModelEvalTarget:
    data = json.loads(payload)
    return TTSModelEvalTarget(
        model_key=str(data["model_key"]),
        profile_key=data.get("profile_key"),
        requested_voice=data.get("requested_voice"),
        voice_source=data.get("voice_source"),
    )


def run_isolated_target(
    target: TTSModelEvalTarget,
    *,
    args: argparse.Namespace,
    audio_dir: Path | None,
    index: int,
    total: int,
) -> dict[str, Any]:
    worker_dir = args.output_dir / "tts_bakeoff" / "_workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    result_path = worker_dir / (
        f"{safe_filename(target.profile_key or 'model')}__{safe_filename(target.model_key)}"
        f"__{time.time_ns()}.json"
    )
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-target-json",
        worker_payload(target),
        "--worker-result",
        str(result_path),
        "--prompts",
        str(args.prompts),
        "--voice-policy",
        args.voice_policy,
        "--output-dir",
        str(args.output_dir),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.max_voices is not None:
        command.extend(["--max-voices", str(args.max_voices)])
    if args.no_write_audio:
        command.append("--no-write-audio")
    elif audio_dir is not None:
        command.extend(["--worker-audio-dir", str(audio_dir)])
    if args.no_skip_unavailable:
        command.append("--no-skip-unavailable")

    label = target.profile_key or target.model_key
    console.print(
        f"\n[bold]({index}/{total}) Isolated TTS benchmark[/bold] "
        f"{target.model_key}"
        f"{f' profile={target.profile_key}' if target.profile_key else ''}"
        f"{f' voice={target.requested_voice}' if target.requested_voice else ''}"
    )
    with console.status(f"Running {label} in a clean subprocess...", spinner="dots"):
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        result_path.unlink(missing_ok=True)
        try:
            worker_dir.rmdir()
        except OSError:
            pass
        result = payload["result"]
        status = result.get("status", "unknown") if result else "none"
        console.print(f"[green]Done[/green] {target.model_key}: {status}")
        return result

    config = get_tts_model_config(target.model_key)
    return {
        "profile": target.profile_key,
        "model": target.model_key,
        "tier": tts_tier(target.model_key),
        "role": config.role,
        "runner": config.runner,
        "status": "worker_failed",
        "skip_reason": (completed.stderr or completed.stdout).strip(),
        "license": config.license,
        "source_url": config.source_url,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Soca TTS bakeoff.")
    parser.add_argument("--model", action="append", default=[], help="TTS model key. Can be comma-separated.")
    parser.add_argument("--profile", action="append", default=[], choices=sorted(VOICE_RUNTIME_PROFILES))
    parser.add_argument("--all", dest="all_models", action="store_true", help="Run every TTS registry candidate.")
    parser.add_argument("--tier-a", action="store_true", help="Run all Tier A registry candidates.")
    parser.add_argument("--tier-b", action="store_true", help="Run all Tier B quality baseline candidates.")
    parser.add_argument(
        "--voice",
        default=None,
        help="Voice/speaker id where supported. Only valid for exactly one selected model/profile.",
    )
    parser.add_argument(
        "--voice-map",
        action="append",
        default=[],
        help="Per-model voice override as MODEL=VOICE. Can be passed multiple times.",
    )
    parser.add_argument("--voice-policy", choices=VOICE_POLICIES, default="default")
    parser.add_argument("--max-voices", type=int, default=None)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-write-audio", action="store_true", help="Do not write generated WAV files.")
    parser.add_argument(
        "--strict-prompts",
        action="store_true",
        help="Fail if the prompt corpus does not cover every required benchmark category.",
    )
    parser.add_argument(
        "--isolate-model-process",
        action="store_true",
        help="Run each selected model/profile in a fresh subprocess for cleaner RAM/load measurements.",
    )
    parser.add_argument(
        "--no-skip-unavailable",
        action="store_true",
        help="Raise instead of skipping registry entries whose runtime is unavailable.",
    )
    parser.add_argument("--worker-target-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-audio-dir", type=Path, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prompts = load_prompts(args.prompts, limit=args.limit)
    prompt_coverage = summarize_prompt_coverage(prompts)
    if args.strict_prompts:
        validate_prompt_coverage(prompts)

    if args.worker_target_json:
        if args.worker_result is None:
            raise ValueError("--worker-result is required with --worker-target-json")
        target = target_from_worker_payload(args.worker_target_json)
        audio_dir = None if args.no_write_audio else args.worker_audio_dir
        result = run_model_eval(
            target,
            prompts,
            voice_policy=args.voice_policy,
            max_voices=args.max_voices,
            skip_unavailable=not args.no_skip_unavailable,
            audio_dir=audio_dir,
        )
        args.worker_result.parent.mkdir(parents=True, exist_ok=True)
        args.worker_result.write_text(
            json.dumps(
                {"prompt_coverage": prompt_coverage, "result": result},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return 0

    model_keys = select_model_keys(
        parse_model_list(args.model),
        tier_a=args.tier_a,
        tier_b=args.tier_b,
        all_models=args.all_models,
    )
    targets = build_eval_targets(
        model_keys,
        args.profile,
        voice=args.voice,
        voice_map=parse_voice_map(args.voice_map),
    )
    console.print(
        f"[bold]TTS bakeoff[/bold]: {len(targets)} target(s), {len(prompts)} prompt(s), "
        f"voice_policy={args.voice_policy}, isolated={args.isolate_model_process}"
    )
    created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_paths = make_eval_run_paths(args.output_dir, "tts_bakeoff", created_at)
    audio_dir = None if args.no_write_audio else run_paths.run_dir / "audio"

    results: list[dict[str, Any]] = []
    for index, target in enumerate(targets, start=1):
        if args.isolate_model_process:
            result = run_isolated_target(
                target,
                args=args,
                audio_dir=audio_dir,
                index=index,
                total=len(targets),
            )
        else:
            result = run_model_eval(
                target,
                prompts,
                voice_policy=args.voice_policy,
                max_voices=args.max_voices,
                skip_unavailable=not args.no_skip_unavailable,
                audio_dir=audio_dir,
            )
        if result is not None:
            results.append(result)

    render_summary(results)
    json_path, md_path = write_outputs(
        results,
        run_paths,
        audio_dir=audio_dir,
        prompt_coverage=prompt_coverage,
    )
    console.print(f"\n[green]Saved[/green] {json_path}")
    console.print(f"[green]Saved[/green] {md_path}")
    if audio_dir is not None:
        console.print(f"[green]Saved audio[/green] {audio_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
