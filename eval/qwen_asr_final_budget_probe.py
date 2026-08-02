from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import numpy as np

from eval.asr_release_config import load_release_config
from eval.asr_release_metrics import error_rates
from eval.asr_release_runner import (
    BenchmarkItem,
    load_audio,
    load_release_items,
    open_qwen_backend,
)
from soca.asr.qwen_ipc_protocol import SAMPLE_RATE
from soca.asr.qwen_service_client import DEFAULT_REQUEST_TIMEOUT_S, QwenServiceTimeout
from soca.asr.vad import SpeechDetector

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "eval/gates/qwen_asr_release.json"


@click.command()
@click.option("--artifact", "artifact_key", required=True)
@click.option("--run-dir", type=click.Path(path_type=Path), required=True)
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=DEFAULT_CONFIG)
def main(artifact_key: str, run_dir: Path, config_path: Path) -> None:
    config = load_release_config(config_path)
    if artifact_key not in config.artifacts:
        raise click.ClickException(f"artifact is outside the release matrix: {artifact_key}")
    config.verify_datasets(REPO_ROOT)
    items = load_release_items(config, REPO_ROOT)["fleurs_vi"]
    longest = sorted(items, key=lambda item: len(load_audio(item.path)), reverse=True)[:4]
    scenarios = _build_scenarios(longest)
    output_path = run_dir.resolve() / f"final-budget-{artifact_key}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_key": artifact_key,
        "status": "running",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "request_timeout_s": config.decode["budget_probe_timeout_s"],
        "production_request_timeout_s": DEFAULT_REQUEST_TIMEOUT_S,
        "fallback_attempted": False,
        "rows": [],
    }
    _write_json(output_path, payload)

    try:
        backend, _identity = open_qwen_backend(
            artifact_key,
            detector=SpeechDetector(),
            request_timeout_s=float(config.decode["budget_probe_timeout_s"]),
        )
        try:
            backend.transcribe(
                scenarios[0][1][:SAMPLE_RATE],
                max_new_tokens=16,
                context="",
            )
            for scenario_name, audio, reference, item_ids in scenarios:
                for max_new_tokens in config.decode["final_budget_candidates"]:
                    row = _run_candidate(
                        backend,
                        scenario_name=scenario_name,
                        audio=audio,
                        reference=reference,
                        item_ids=item_ids,
                        max_new_tokens=max_new_tokens,
                    )
                    payload["rows"].append(row)
                    _write_json(output_path, payload)
        finally:
            backend.close()
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


def _run_candidate(
    backend: Any,
    *,
    scenario_name: str,
    audio: np.ndarray,
    reference: str,
    item_ids: list[str],
    max_new_tokens: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scenario": scenario_name,
        "item_ids": item_ids,
        "max_new_tokens": max_new_tokens,
        "audio_duration_ms": len(audio) / SAMPLE_RATE * 1_000,
        "fallback_attempted": False,
    }
    try:
        started = time.perf_counter()
        result = backend.transcribe(
            audio,
            max_new_tokens=max_new_tokens,
            context="",
        )
        wall_ms = (time.perf_counter() - started) * 1_000
        rates = error_rates([reference], [result.text])
        row.update(
            {
                "status": "completed",
                "wall_ms": wall_ms,
                "production_deadline_met": wall_ms <= DEFAULT_REQUEST_TIMEOUT_S * 1_000,
                "rtf": result.rtf,
                "wer": rates.wer,
                "reference_words": rates.reference_words,
                "hypothesis_words": len(result.text.split()),
                "generated_token_count": result.generated_token_count,
                "hit_max_new_tokens": result.hit_max_new_tokens,
            }
        )
    except QwenServiceTimeout as exc:
        row.update(
            {
                "status": "timeout",
                "production_deadline_met": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
    return row


def _build_scenarios(
    items: list[BenchmarkItem],
) -> list[tuple[str, np.ndarray, str, list[str]]]:
    silence = np.zeros(4_000, dtype=np.float32)
    scenarios = []
    for item_count in (1, 2, 4):
        selected = items[:item_count]
        parts: list[np.ndarray] = []
        for index, item in enumerate(selected):
            if index:
                parts.append(silence)
            parts.append(load_audio(item.path))
        scenarios.append(
            (
                f"joined_{item_count}",
                np.concatenate(parts),
                " ".join(item.reference for item in selected),
                [item.item_id for item in selected],
            )
        )
    return scenarios


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
