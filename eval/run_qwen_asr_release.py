from __future__ import annotations

import gc
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from eval.asr_release_config import load_release_config
from eval.asr_release_runner import (
    GuardThresholds,
    RawPrediction,
    build_context_factory,
    calibration_record,
    collect_predictions,
    context_echo_screening,
    derive_thresholds,
    evaluate_release_quality_gates,
    load_release_items,
    new_run_id,
    open_qwen_backend,
    runtime_metadata,
    summarize_predictions,
    write_calibration_artifact,
)
from soca.asr.calibration import QWEN_CONFIDENCE_CALIBRATION_PATH
from soca.asr.robust_asr import load_confidence_guard_calibration
from soca.asr.vad import SpeechDetector
from soca.asr.voice_backend import PhoWhisperVoiceBackend
from soca.core.knowledge_setup import build_knowledge_runtime_setup
from soca.knowledge.factory import RetrievalConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "eval/gates/qwen_asr_release.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "eval/results/qwen_asr_release"


@click.group()
def main() -> None:
    pass


@main.command("validate")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=DEFAULT_CONFIG)
def validate(config_path: Path) -> None:
    config = load_release_config(config_path)
    digests = config.verify_datasets(REPO_ROOT)
    items = load_release_items(config, REPO_ROOT)
    click.echo(
        json.dumps(
            {
                "config_digest": config.digest,
                "dataset_digests": digests,
                "item_counts": {name: len(rows) for name, rows in items.items()},
            },
            indent=2,
        )
    )


@main.command("run")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=DEFAULT_CONFIG)
@click.option("--vault", type=click.Path(path_type=Path), required=True)
@click.option("--output-root", type=click.Path(path_type=Path), default=DEFAULT_OUTPUT_ROOT)
@click.option("--resume-run", type=click.Path(path_type=Path), default=None)
@click.option(
    "--calibration-output",
    type=click.Path(path_type=Path),
    default=QWEN_CONFIDENCE_CALIBRATION_PATH,
)
def run(
    config_path: Path,
    vault: Path,
    output_root: Path,
    resume_run: Path | None,
    calibration_output: Path,
) -> None:
    config = load_release_config(config_path)
    dataset_digests = config.verify_datasets(REPO_ROOT)
    items_by_dataset = load_release_items(config, REPO_ROOT)
    all_items = [
        item
        for dataset_name in ("fleurs_vi", "non_speech", "private_codeswitch")
        for item in items_by_dataset[dataset_name]
    ]
    if resume_run is None:
        run_id = new_run_id()
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        metadata = runtime_metadata(config, dataset_digests)
        metadata.update({"run_id": run_id, "status": "running", "vault": str(vault.resolve())})
    else:
        run_dir = resume_run.resolve()
        metadata = _load_run_metadata(run_dir / "run.json")
        _verify_resume(metadata, config.digest, dataset_digests, vault.resolve())
        run_id = str(metadata["run_id"])
        metadata.update(
            {
                "status": "running",
                "resumed_at_utc": datetime.now(UTC).isoformat(),
            }
        )
        for stale_key in ("completed_at_utc", "error_type", "error_message"):
            metadata.pop(stale_key, None)
    _write_json(run_dir / "run.json", metadata)

    setup = None
    rows_by_artifact: dict[str, list[RawPrediction]] = {}
    summaries: dict[str, Any] = {}
    calibration_records: dict[str, dict[str, Any]] = {}
    try:
        setup = build_knowledge_runtime_setup(
            vault.resolve(),
            knowledge_limit=5,
            retrieval_config=RetrievalConfig(mode="cached_sparse"),
        )
        context_factory = build_context_factory(setup.catalog)
        detector = SpeechDetector()
        control_calibration = load_confidence_guard_calibration("phowhisper_small")
        if control_calibration is None:
            raise RuntimeError("PhoWhisper evaluation control calibration is missing")
        click.echo("Starting explicit PhoWhisper evaluation control")
        control_path = run_dir / "phowhisper_small_control.jsonl"
        existing_control = _load_rows(control_path)
        control_backend = (
            PhoWhisperVoiceBackend("phowhisper_small")
            if len(existing_control) < len(all_items)
            else None
        )
        try:
            control_rows = collect_predictions(
                items=all_items,
                backend=control_backend or _CheckpointOnlyBackend(),
                detector=detector,
                context_variants=("empty",),
                context_factory=context_factory,
                max_new_tokens=int(config.decode["max_new_tokens"]),
                progress=_progress,
                existing_rows=existing_control,
                row_sink=_row_appender(control_path),
            )
        finally:
            if control_backend is not None:
                control_backend.close()
                del control_backend
                gc.collect()
        rows_by_artifact["phowhisper_small_control"] = control_rows
        _write_rows(control_path, control_rows)
        summaries["phowhisper_small_control"] = summarize_predictions(
            control_rows,
            thresholds=GuardThresholds(
                min_avg_logprob=control_calibration.min_avg_logprob,
                max_compression_ratio=control_calibration.max_compression_ratio,
                context_echo_min_contiguous_tokens=int(
                    config.context_echo_candidates["selected_min_contiguous_tokens"]
                ),
            ),
        )
        for artifact_key in config.artifacts:
            click.echo(f"Starting {artifact_key} through the production Qwen service")
            artifact_path = run_dir / f"{artifact_key}.jsonl"
            existing_rows = _load_rows(artifact_path)
            backend, identity = open_qwen_backend(artifact_key, detector=detector)
            try:
                rows = collect_predictions(
                    items=all_items,
                    backend=backend,
                    detector=detector,
                    context_variants=config.context_variants,
                    context_factory=context_factory,
                    max_new_tokens=int(config.decode["max_new_tokens"]),
                    progress=_progress,
                    existing_rows=existing_rows,
                    row_sink=_row_appender(artifact_path),
                )
            finally:
                backend.close()
            rows_by_artifact[artifact_key] = rows
            _write_rows(artifact_path, rows)
            thresholds = derive_thresholds(
                rows,
                max_speech_false_reject_rate=float(
                    config.threshold_selection["max_calibration_speech_false_reject_rate"]
                ),
                compression_floor=float(config.threshold_selection["max_compression_ratio_floor"]),
                context_echo_min_contiguous_tokens=int(
                    config.context_echo_candidates["selected_min_contiguous_tokens"]
                ),
            )
            summaries[artifact_key] = summarize_predictions(rows, thresholds)
            summaries[artifact_key]["context_echo_screening"] = context_echo_screening(
                rows,
                thresholds=[
                    float(value)
                    for value in config.context_echo_candidates["token_overlap_thresholds"]
                ],
                minimum_unique_tokens=int(config.context_echo_candidates["minimum_unique_tokens"]),
                contiguous_thresholds=tuple(
                    int(value)
                    for value in config.context_echo_candidates["contiguous_token_thresholds"]
                ),
            )
            calibration_records[identity.digest] = calibration_record(
                identity=identity,
                thresholds=thresholds,
                config=config,
                dataset_digests=dataset_digests,
                rows=rows,
                run_id=run_id,
            )

        write_calibration_artifact(calibration_output, calibration_records)
        summaries["release_gate_evaluation"] = evaluate_release_quality_gates(
            summaries,
            release_key="qwen3_asr_0_6b",
            control_key="phowhisper_small_control",
            gates=config.release_gates,
        )
        _write_json(run_dir / "summary.json", summaries)
        metadata.update(
            {
                "status": "completed",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "calibration_output": str(calibration_output.resolve()),
                "artifacts": list(config.artifacts),
            }
        )
        _write_json(run_dir / "run.json", metadata)
        click.echo(f"Completed raw release run: {run_dir}")
    except BaseException as exc:
        metadata.update(
            {
                "status": "failed",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
        _write_json(run_dir / "run.json", metadata)
        raise
    finally:
        if setup is not None:
            close = getattr(setup.source, "close", None)
            if callable(close):
                close()


def _progress(completed: int, total: int, item, variant: str) -> None:
    if completed not in {1, total} and completed % 25 != 0:
        return
    click.echo(
        f"[{completed}/{total}] {item.dataset}:{item.item_id} context={variant}",
        err=True,
    )


def _write_rows(path: Path, rows: list[RawPrediction]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def _row_appender(path: Path) -> Any:
    def append(row: RawPrediction) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
            handle.flush()

    return append


def _load_rows(path: Path) -> list[RawPrediction]:
    if not path.exists():
        return []
    rows: list[RawPrediction] = []
    expected = set(RawPrediction.__dataclass_fields__)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict) or set(payload) != expected:
                    raise ValueError("fields do not match RawPrediction")
                payload["context_provenance"] = tuple(payload["context_provenance"])
                payload["english_reference_indices"] = tuple(payload["english_reference_indices"])
                rows.append(RawPrediction(**payload))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise click.ClickException(
                    f"invalid checkpoint row {path}:{line_number}: {exc}"
                ) from exc
    return rows


def _load_run_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"cannot read resume metadata: {path}") from exc
    if not isinstance(payload, dict):
        raise click.ClickException("resume metadata must be a JSON object")
    return payload


def _verify_resume(
    metadata: dict[str, Any],
    config_digest: str,
    dataset_digests: dict[str, str],
    vault: Path,
) -> None:
    expected = {
        "benchmark_config_digest": config_digest,
        "datasets": dataset_digests,
        "vault": str(vault),
    }
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise click.ClickException("resume identity mismatch: " + ", ".join(sorted(mismatches)))


class _CheckpointOnlyBackend:
    def transcribe(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("complete checkpoint unexpectedly requested inference")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
