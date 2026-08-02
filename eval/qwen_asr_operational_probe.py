from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import click

from eval.asr_release_config import load_release_config
from eval.asr_release_metrics import percentile
from eval.asr_release_runner import (
    BenchmarkItem,
    build_context_factory,
    load_audio,
    load_release_items,
    open_qwen_backend,
)
from soca.asr.calibration import QWEN_ASR_PARTIAL_MAX_NEW_TOKENS
from soca.asr.qwen_artifacts import default_asr_model_root, get_qwen_artifact
from soca.asr.qwen_service_client import QwenServiceCrashed
from soca.core.knowledge_setup import build_knowledge_runtime_setup
from soca.knowledge.factory import RetrievalConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "eval/gates/qwen_asr_release.json"


@click.command()
@click.option("--artifact", "artifact_key", required=True)
@click.option("--vault", type=click.Path(path_type=Path), required=True)
@click.option("--run-dir", type=click.Path(path_type=Path), required=True)
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=DEFAULT_CONFIG)
def main(artifact_key: str, vault: Path, run_dir: Path, config_path: Path) -> None:
    config = load_release_config(config_path)
    if artifact_key not in config.artifacts:
        raise click.ClickException(f"artifact is outside the release matrix: {artifact_key}")
    config.verify_datasets(REPO_ROOT)
    items = load_release_items(config, REPO_ROOT)["fleurs_vi"]
    buckets = _duration_bucket_items(items)
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / f"operational-{artifact_key}.json"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_key": artifact_key,
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "fallback_attempted": False,
        "start_stop": [],
    }
    _write_json(output_path, payload)

    setup = build_knowledge_runtime_setup(
        vault.resolve(),
        knowledge_limit=5,
        retrieval_config=RetrievalConfig(mode="cached_sparse"),
    )
    context = build_context_factory(setup.catalog)("production_catalog")
    representative = load_audio(buckets["medium"].path)
    try:
        for repetition in range(int(config.repetitions["start_stop"])):
            started = time.perf_counter()
            backend, identity = open_qwen_backend(artifact_key, detector=_detector())
            startup_ms = (time.perf_counter() - started) * 1_000
            process = backend.process
            if process is None:
                raise RuntimeError("Qwen client has no worker process")
            row: dict[str, Any] = {
                "repetition": repetition,
                "startup_ms": startup_ms,
                "ready_rss_mb": _rss_mb(process.pid),
                "identity_digest": identity.digest,
                "no_fallback_attempted": bool(
                    backend.identity and backend.identity.no_fallback_attempted
                ),
            }
            if repetition < int(config.repetitions["cold_process"]):
                wall_started = time.perf_counter()
                result = backend.transcribe(representative, context=context.text)
                wall_ms = (time.perf_counter() - wall_started) * 1_000
                row["cold_inference"] = {
                    "wall_ms": wall_ms,
                    "backend_ms": result.latency_ms,
                    "ipc_overhead_ms": max(0.0, wall_ms - result.latency_ms),
                    "rtf": result.rtf,
                    "worker_rss_mb": _rss_mb(process.pid),
                }
            socket_path = backend.socket_path
            ready_path = backend.ready_path
            backend.close()
            row.update(
                {
                    "exit_code": process.poll(),
                    "socket_removed": not socket_path.exists(),
                    "ready_marker_removed": not ready_path.exists(),
                }
            )
            payload["start_stop"].append(row)
            _write_json(output_path, payload)

        backend, identity = open_qwen_backend(artifact_key, detector=_detector())
        try:
            process = backend.process
            if process is None:
                raise RuntimeError("Qwen client has no worker process")
            payload["warm"] = _warm_probe(
                backend,
                buckets,
                context.text,
                repetitions=int(config.repetitions["warm_per_item"]),
            )
            payload["concurrency"] = _concurrency_probe(
                backend,
                load_audio(buckets["medium"].path),
                context.text,
            )
            payload["peak_observed_worker_rss_mb"] = _rss_mb(process.pid)
            spec = get_qwen_artifact(artifact_key)
            payload["disk_bytes"] = _tree_bytes(
                default_asr_model_root() / spec.key / spec.upstream.revision
            )
            process.kill()
            process.wait(timeout=5)
            try:
                backend.transcribe(representative, context=context.text)
            except QwenServiceCrashed as exc:
                payload["crash_probe"] = {
                    "passed": True,
                    "error_type": type(exc).__name__,
                    "fallback_attempted": False,
                }
            else:
                raise RuntimeError("killed Qwen worker did not raise QwenServiceCrashed")
        finally:
            backend.close()

        payload["summary"] = _summarize(payload)
        payload["status"] = "completed"
        payload["completed_at_utc"] = datetime.now(UTC).isoformat()
        _write_json(output_path, payload)
        click.echo(output_path)
    except BaseException as exc:
        payload.update(
            {
                "status": "failed",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        _write_json(output_path, payload)
        raise
    finally:
        close = getattr(setup.source, "close", None)
        if callable(close):
            close()


def _detector():
    from soca.asr.vad import SpeechDetector

    return SpeechDetector()


def _duration_bucket_items(items: list[BenchmarkItem]) -> dict[str, BenchmarkItem]:
    import soundfile as sf

    ranked = sorted(items, key=lambda item: sf.info(item.path).duration)
    return {
        "short": ranked[len(ranked) // 10],
        "medium": ranked[len(ranked) // 2],
        "long": ranked[(len(ranked) * 9) // 10],
    }


def _warm_probe(
    backend: Any,
    buckets: dict[str, BenchmarkItem],
    context: str,
    *,
    repetitions: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, item in buckets.items():
        audio = load_audio(item.path)
        finals = []
        for _ in range(repetitions):
            started = time.perf_counter()
            result = backend.transcribe(audio, context=context)
            wall_ms = (time.perf_counter() - started) * 1_000
            finals.append(
                {
                    "wall_ms": wall_ms,
                    "backend_ms": result.latency_ms,
                    "ipc_overhead_ms": max(0.0, wall_ms - result.latency_ms),
                    "rtf": result.rtf,
                }
            )
        partials = []
        for fraction in (0.25, 0.5, 0.75):
            prefix = audio[: max(1, int(len(audio) * fraction))]
            for repetition in range(repetitions):
                started = time.perf_counter()
                result = backend.transcribe(
                    prefix,
                    max_new_tokens=QWEN_ASR_PARTIAL_MAX_NEW_TOKENS,
                    context="",
                )
                partials.append(
                    {
                        "fraction": fraction,
                        "repetition": repetition,
                        "max_new_tokens": QWEN_ASR_PARTIAL_MAX_NEW_TOKENS,
                        "wall_ms": (time.perf_counter() - started) * 1_000,
                        "backend_ms": result.latency_ms,
                        "text": result.text,
                        "generated_token_count": result.generated_token_count,
                        "hit_max_new_tokens": result.hit_max_new_tokens,
                    }
                )
        report[name] = {
            "item_id": item.item_id,
            "audio_duration_ms": len(audio) / 16,
            "final_repetitions": finals,
            "partials": partials,
        }
    return report


def _concurrency_probe(backend: Any, audio: Any, context: str) -> dict[str, Any]:
    def partial() -> dict[str, Any]:
        started = time.perf_counter()
        result = backend.transcribe(
            audio[: len(audio) // 2],
            max_new_tokens=QWEN_ASR_PARTIAL_MAX_NEW_TOKENS,
            context="",
        )
        return {"wall_ms": (time.perf_counter() - started) * 1_000, **asdict(result)}

    def final() -> dict[str, Any]:
        started = time.perf_counter()
        result = backend.transcribe(audio, context=context)
        return {"wall_ms": (time.perf_counter() - started) * 1_000, **asdict(result)}

    with ThreadPoolExecutor(max_workers=2) as pool:
        partial_future = pool.submit(partial)
        final_future = pool.submit(final)
        return {
            "partial": partial_future.result(),
            "final": final_future.result(),
            "failures": 0,
        }


def _summarize(payload: dict[str, Any]) -> dict[str, Any]:
    starts = [float(row["startup_ms"]) for row in payload["start_stop"]]
    cold = [row["cold_inference"] for row in payload["start_stop"] if "cold_inference" in row]
    partial_ms = [
        float(partial["wall_ms"])
        for bucket in payload["warm"].values()
        for partial in bucket["partials"]
    ]
    repeat_groups = [
        [partial["text"] for partial in bucket["partials"] if partial["fraction"] == fraction]
        for bucket in payload["warm"].values()
        for fraction in (0.25, 0.5, 0.75)
    ]
    adjacent_similarity = []
    for bucket in payload["warm"].values():
        first_by_fraction = [
            next(
                partial["text"]
                for partial in bucket["partials"]
                if partial["fraction"] == fraction and partial["repetition"] == 0
            )
            for fraction in (0.25, 0.5, 0.75)
        ]
        adjacent_similarity.extend(
            SequenceMatcher(None, left.casefold().split(), right.casefold().split()).ratio()
            for left, right in zip(first_by_fraction, first_by_fraction[1:], strict=False)
        )
    cleanup_violations = sum(
        not row["socket_removed"] or not row["ready_marker_removed"]
        for row in payload["start_stop"]
    )
    fallback_attempts = sum(
        not row["no_fallback_attempted"] for row in payload["start_stop"]
    )
    return {
        "start_stop_count": len(starts),
        "start_stop_failures": sum(
            row["exit_code"] != 0
            or not row["socket_removed"]
            or not row["ready_marker_removed"]
            or not row["no_fallback_attempted"]
            for row in payload["start_stop"]
        ),
        "startup_ms_median": percentile(starts, 0.5),
        "startup_ms_p95": percentile(starts, 0.95),
        "cold_inference_wall_ms_p95": percentile([float(row["wall_ms"]) for row in cold], 0.95),
        "ipc_overhead_ms_p95": percentile([float(row["ipc_overhead_ms"]) for row in cold], 0.95),
        "partial_wall_ms_p95": percentile(partial_ms, 0.95),
        "partial_repeat_consistency_rate": sum(len(set(group)) == 1 for group in repeat_groups)
        / len(repeat_groups),
        "partial_adjacent_similarity_mean": sum(adjacent_similarity) / len(adjacent_similarity),
        "orphan_process_count": cleanup_violations,
        "fallback_attempt_count": fallback_attempts,
    }


def _rss_mb(pid: int) -> float:
    output = subprocess.check_output(
        ["ps", "-o", "rss=", "-p", str(pid)],
        text=True,
    ).strip()
    if not output:
        raise RuntimeError(f"cannot read RSS for worker {pid}")
    return int(output) / 1024


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(entry.stat().st_size for entry in path.rglob("*") if entry.is_file())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
