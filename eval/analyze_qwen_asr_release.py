from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import click

from eval.asr_release_config import load_release_config
from eval.asr_release_runner import (
    GuardThresholds,
    ReleaseBenchmarkError,
    calibration_record,
    context_echo_screening,
    derive_thresholds,
    evaluate_context_echo_labels,
    evaluate_release_quality_gates,
    summarize_predictions,
    write_calibration_artifact,
)
from eval.run_qwen_asr_release import _load_rows, _write_json
from soca.asr.calibration import (
    QWEN_CONFIDENCE_CALIBRATION_PATH,
    compute_vad_policy_digest,
    qwen_calibration_identity,
)
from soca.asr.qwen_artifacts import get_qwen_artifact
from soca.asr.robust_asr import load_confidence_guard_calibration
from soca.asr.vad import SpeechDetector

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "eval/gates/qwen_asr_release.json"
DEFAULT_LABELS = REPO_ROOT / "eval/labels/qwen_context_echo_release.json"


@click.command()
@click.option("--run-dir", type=click.Path(path_type=Path), required=True)
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=DEFAULT_CONFIG)
@click.option("--labels", "labels_path", type=click.Path(path_type=Path), default=DEFAULT_LABELS)
@click.option(
    "--calibration-output",
    type=click.Path(path_type=Path),
    default=QWEN_CONFIDENCE_CALIBRATION_PATH,
)
def main(
    run_dir: Path,
    config_path: Path,
    labels_path: Path,
    calibration_output: Path,
) -> None:
    run_dir = run_dir.resolve()
    metadata = _read_object(run_dir / "run.json")
    config = load_release_config(config_path)
    dataset_digests = config.verify_datasets(REPO_ROOT)
    if metadata.get("status") != "completed":
        raise click.ClickException("release run must be completed before analysis")
    if metadata.get("datasets") != dataset_digests:
        raise click.ClickException("analysis dataset identity differs from the raw run")
    inference_config = run_dir / "benchmark_config.json"
    if _sha256(inference_config) != metadata.get("benchmark_config_digest"):
        raise click.ClickException("immutable inference config digest does not match run metadata")

    labels = _read_object(labels_path)
    if labels.get("source_run_id") != metadata.get("run_id"):
        raise click.ClickException("manual labels belong to a different raw run")
    if labels.get("source_benchmark_config_digest") != metadata.get("benchmark_config_digest"):
        raise click.ClickException("manual labels belong to a different inference config")
    selected = labels.get("selected_policy")
    threshold = int(config.context_echo_candidates["selected_min_contiguous_tokens"])
    if not isinstance(selected, dict) or selected != {
        "algorithm": "contiguous_context_span_v1",
        "minimum_contiguous_tokens": threshold,
    }:
        raise click.ClickException("manual label policy differs from analysis config")

    control_rows = _load_rows(run_dir / "phowhisper_small_control.jsonl")
    control_calibration = load_confidence_guard_calibration("phowhisper_small")
    if control_calibration is None:
        raise click.ClickException("PhoWhisper evaluation control calibration is missing")
    summaries: dict[str, Any] = {
        "phowhisper_small_control": summarize_predictions(
            control_rows,
            GuardThresholds(
                control_calibration.min_avg_logprob,
                control_calibration.max_compression_ratio,
                threshold,
            ),
        )
    }
    records: dict[str, dict[str, Any]] = {}
    detector = SpeechDetector()
    artifacts = labels.get("artifacts")
    if not isinstance(artifacts, dict):
        raise click.ClickException("manual labels have no artifact records")
    for artifact_key in config.artifacts:
        rows = _load_rows(run_dir / f"{artifact_key}.jsonl")
        thresholds = derive_thresholds(
            rows,
            max_speech_false_reject_rate=float(
                config.threshold_selection["max_calibration_speech_false_reject_rate"]
            ),
            compression_floor=float(config.threshold_selection["max_compression_ratio_floor"]),
            context_echo_min_contiguous_tokens=threshold,
        )
        summary = summarize_predictions(rows, thresholds)
        summary["context_echo_screening"] = context_echo_screening(
            rows,
            thresholds=[
                float(value) for value in config.context_echo_candidates["token_overlap_thresholds"]
            ],
            minimum_unique_tokens=int(config.context_echo_candidates["minimum_unique_tokens"]),
            contiguous_thresholds=tuple(
                int(value)
                for value in config.context_echo_candidates["contiguous_token_thresholds"]
            ),
        )
        artifact_labels = artifacts.get(artifact_key)
        if not isinstance(artifact_labels, dict) or not isinstance(
            artifact_labels.get("labels"), list
        ):
            raise click.ClickException(f"manual labels are missing for {artifact_key}")
        summary["context_echo_manual_review"] = evaluate_context_echo_labels(
            rows,
            artifact_labels["labels"],
            minimum_contiguous_tokens=threshold,
        )
        summaries[artifact_key] = summary
        identity = qwen_calibration_identity(
            get_qwen_artifact(artifact_key),
            vad_policy_digest=compute_vad_policy_digest(detector),
        )
        records[identity.digest] = calibration_record(
            identity=identity,
            thresholds=thresholds,
            config=config,
            dataset_digests=dataset_digests,
            rows=rows,
            run_id=str(metadata["run_id"]),
        )

    summaries["release_gate_evaluation"] = evaluate_release_quality_gates(
        summaries,
        release_key="qwen3_asr_0_6b",
        control_key="phowhisper_small_control",
        gates=config.release_gates,
    )
    summaries["analysis_provenance"] = {
        "inference_config_digest": metadata["benchmark_config_digest"],
        "analysis_config_digest": config.digest,
        "manual_label_digest": _sha256(labels_path),
        "raw_run_id": metadata["run_id"],
    }
    write_calibration_artifact(calibration_output, records)
    _write_json(run_dir / "summary.json", summaries)
    metadata["benchmark_config"] = str(inference_config)
    metadata["analysis"] = summaries["analysis_provenance"]
    _write_json(run_dir / "run.json", metadata)
    click.echo(json.dumps(summaries["release_gate_evaluation"], indent=2))


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"cannot read JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise click.ClickException(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ReleaseBenchmarkError(f"evidence file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
