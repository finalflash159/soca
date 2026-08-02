from __future__ import annotations

import json
import math
import platform
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import librosa
import numpy as np
import soundfile as sf

from eval.asr_release_config import ASRReleaseConfig
from eval.asr_release_metrics import (
    code_switch_metrics,
    error_rates,
    percentile,
    ranking_metrics,
    select_lower_bound_threshold,
    wilson_interval,
)
from local.codeswitch_text import english_indices, normalize
from soca.asr.calibration import (
    ASRCalibrationIdentity,
    compute_vad_policy_digest,
    qwen_calibration_identity,
)
from soca.asr.context import ASRContextBuilder, ASRContextSnapshot
from soca.asr.context_sources import runtime_context_records
from soca.asr.hallucination_heuristics import (
    compression_ratio,
    looks_like_context_echo,
    max_contiguous_context_tokens,
)
from soca.asr.protocols import VoiceASRBackend
from soca.asr.qwen_artifacts import default_asr_model_root, get_qwen_artifact
from soca.asr.qwen_service_client import DEFAULT_REQUEST_TIMEOUT_S, QwenASRServiceClient
from soca.asr.qwen_service_identity import QwenServiceLaunch
from soca.asr.qwen_store import QwenArtifactStore
from soca.asr.result import ASRResult
from soca.asr.vad import SpeechDetector

SAMPLE_RATE = 16_000


class ReleaseBenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BenchmarkItem:
    item_id: str
    dataset: str
    path: Path
    reference: str
    cohort: str
    english_reference_indices: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class RawPrediction:
    item_id: str
    dataset: str
    cohort: str
    context_variant: str
    context_digest: str
    context_provenance: tuple[str, ...]
    reference: str
    text: str
    vad_has_speech: bool
    speech_duration_ms: float
    vad_latency_ms: float
    asr_latency_ms: float
    audio_duration_ms: float
    rtf: float
    avg_logprob: float | None
    avg_logprob_reliable: bool
    compression_ratio: float
    context_unique_token_overlap: float
    context_max_contiguous_tokens: int
    exact_context_echo: bool
    context_echo_rejected: bool
    english_reference_indices: tuple[int, ...]
    generated_token_count: int | None = None
    hit_max_new_tokens: bool | None = None


@dataclass(frozen=True, slots=True)
class GuardThresholds:
    min_avg_logprob: float
    max_compression_ratio: float
    context_echo_min_contiguous_tokens: int


class ContextFactory(Protocol):
    def __call__(self, variant: str) -> ASRContextSnapshot: ...


def load_release_items(config: ASRReleaseConfig, root: Path) -> dict[str, list[BenchmarkItem]]:
    items: dict[str, list[BenchmarkItem]] = {}
    for dataset_name, contract in config.datasets.items():
        manifest = root / contract.manifest
        rows = _read_jsonl(manifest)
        loaded: list[BenchmarkItem] = []
        for index, row in enumerate(rows):
            if dataset_name == "private_codeswitch":
                item_id = _required_string(row, "id")
                path = _resolve_audio_path(root, manifest, _required_string(row, "wav"))
                reference = _required_string(row, "reference")
                stored_indices = row.get("english_indices")
                expected_indices = english_indices(reference)
                if stored_indices != expected_indices:
                    raise ReleaseBenchmarkError(
                        f"private code-switch term indices are stale for {item_id}"
                    )
                indices = tuple(expected_indices)
                cohort = "holdout"
            elif dataset_name == "fleurs_vi":
                filename = _required_string(row, "filename")
                transcript_id = str(row.get("fleurs_id") or row.get("id") or f"fleurs-{index}")
                item_id = f"{transcript_id}:{filename}"
                nested = manifest.parent / "wav" / filename
                path = nested if nested.is_file() else _resolve_audio_path(root, manifest, filename)
                reference = _required_string(row, "ground_truth")
                indices = ()
                # FLEURS can contain multiple recordings of the same transcript ID.
                # Keep those recordings in one cohort while retaining a unique row ID.
                cohort = config.split_policy.cohort(transcript_id, seed=config.seed)
            else:
                relative_path = _required_string(row, "path")
                item_id = f"{row.get('source', 'noise')}:{relative_path}"
                path = _resolve_audio_path(root, manifest, relative_path)
                reference = ""
                indices = ()
                cohort = config.split_policy.cohort(item_id, seed=config.seed)
            if not path.is_file():
                raise ReleaseBenchmarkError(f"audio file is missing: {path}")
            loaded.append(
                BenchmarkItem(
                    item_id=item_id,
                    dataset=dataset_name,
                    path=path,
                    reference=reference,
                    cohort=cohort,
                    english_reference_indices=indices,
                )
            )
        items[dataset_name] = loaded
    return items


def build_context_factory(knowledge_catalog: Any | None) -> ContextFactory:
    builder = ASRContextBuilder()
    empty = builder.build(())
    catalog = builder.build(runtime_context_records(knowledge_catalog, None))

    def factory(variant: str) -> ASRContextSnapshot:
        if variant == "empty":
            return empty
        if variant == "production_catalog":
            return catalog
        raise ReleaseBenchmarkError(f"unknown context variant: {variant}")

    return factory


def collect_predictions(
    *,
    items: Sequence[BenchmarkItem],
    backend: VoiceASRBackend,
    detector: SpeechDetector,
    context_variants: Sequence[str],
    context_factory: ContextFactory,
    max_new_tokens: int,
    progress: Callable[[int, int, BenchmarkItem, str], None] | None = None,
    existing_rows: Sequence[RawPrediction] = (),
    row_sink: Callable[[RawPrediction], None] | None = None,
) -> list[RawPrediction]:
    total = len(items) * len(context_variants)
    completed = 0
    expected_keys = {
        (item.dataset, item.item_id, variant) for item in items for variant in context_variants
    }
    existing_by_key: dict[tuple[str, str, str], RawPrediction] = {}
    for row in existing_rows:
        key = prediction_key(row)
        if key not in expected_keys:
            raise ReleaseBenchmarkError(f"checkpoint row is outside this run: {key}")
        if key in existing_by_key:
            raise ReleaseBenchmarkError(f"checkpoint contains a duplicate row: {key}")
        existing_by_key[key] = row
    rows: list[RawPrediction] = []
    for item in items:
        missing_variants = [
            variant
            for variant in context_variants
            if (item.dataset, item.item_id, variant) not in existing_by_key
        ]
        audio = None
        vad = None
        if missing_variants:
            audio = load_audio(item.path)
            vad = detector.detect(audio)
        for variant in context_variants:
            completed += 1
            if progress is not None:
                progress(completed, total, item, variant)
            key = (item.dataset, item.item_id, variant)
            checkpoint = existing_by_key.get(key)
            if checkpoint is not None:
                rows.append(checkpoint)
                continue
            if audio is None or vad is None:
                raise AssertionError("missing audio for an uncheckpointed prediction")
            snapshot = context_factory(variant)
            result = (
                backend.transcribe(
                    vad.speech_audio,
                    max_new_tokens=max_new_tokens,
                    context=snapshot.text,
                )
                if vad.has_speech
                else ASRResult(
                    text="",
                    latency_ms=0.0,
                    audio_duration_ms=0.0,
                    rtf=0.0,
                    avg_logprob=0.0,
                    avg_logprob_reliable=False,
                )
            )
            row = RawPrediction(
                item_id=item.item_id,
                dataset=item.dataset,
                cohort=item.cohort,
                context_variant=variant,
                context_digest=snapshot.digest,
                context_provenance=snapshot.provenances,
                reference=item.reference,
                text=result.text,
                vad_has_speech=vad.has_speech,
                speech_duration_ms=vad.speech_duration_ms,
                vad_latency_ms=vad.vad_latency_ms,
                asr_latency_ms=result.latency_ms,
                audio_duration_ms=len(audio) / SAMPLE_RATE * 1000,
                rtf=result.rtf,
                avg_logprob=(result.avg_logprob if vad.has_speech else None),
                avg_logprob_reliable=(result.avg_logprob_reliable and vad.has_speech),
                compression_ratio=compression_ratio(result.text),
                context_unique_token_overlap=_context_token_overlap(result.text, snapshot.text),
                context_max_contiguous_tokens=max_contiguous_context_tokens(
                    result.text, snapshot.text
                ),
                exact_context_echo=_exact_context_echo(result.text, snapshot.text),
                context_echo_rejected=looks_like_context_echo(
                    result.text,
                    snapshot.text,
                    minimum_contiguous_tokens=4,
                ),
                english_reference_indices=item.english_reference_indices,
                generated_token_count=result.generated_token_count,
                hit_max_new_tokens=result.hit_max_new_tokens,
            )
            rows.append(row)
            if row_sink is not None:
                row_sink(row)
    return rows


def prediction_key(row: RawPrediction) -> tuple[str, str, str]:
    return row.dataset, row.item_id, row.context_variant


def derive_thresholds(
    rows: Sequence[RawPrediction],
    *,
    max_speech_false_reject_rate: float,
    compression_floor: float,
    context_echo_min_contiguous_tokens: int,
) -> GuardThresholds:
    speech = [
        row
        for row in rows
        if row.dataset == "fleurs_vi" and row.cohort == "calibration" and row.vad_has_speech
    ]
    logprobs = [
        float(row.avg_logprob)
        for row in speech
        if row.avg_logprob is not None and row.avg_logprob_reliable
    ]
    compression = [row.compression_ratio for row in speech if row.text.strip()]
    if not logprobs or not compression:
        raise ReleaseBenchmarkError("calibration speech produced no reliable ASR scores")
    return GuardThresholds(
        min_avg_logprob=select_lower_bound_threshold(
            logprobs,
            max_false_reject_rate=max_speech_false_reject_rate,
        ),
        max_compression_ratio=max(compression_floor, percentile(compression, 0.99) * 1.15),
        context_echo_min_contiguous_tokens=context_echo_min_contiguous_tokens,
    )


def guard_rejection(row: RawPrediction, thresholds: GuardThresholds) -> str | None:
    if not row.vad_has_speech:
        return "no_speech"
    if row.hit_max_new_tokens is True:
        return "decode_limit_reached"
    if not row.text.strip():
        return "empty_asr"
    if (
        row.avg_logprob is not None
        and row.avg_logprob_reliable
        and row.avg_logprob < thresholds.min_avg_logprob
    ):
        return "low_confidence"
    if row.compression_ratio > thresholds.max_compression_ratio:
        return "high_compression"
    if row.context_max_contiguous_tokens >= thresholds.context_echo_min_contiguous_tokens:
        return "context_echo"
    return None


def summarize_predictions(
    rows: Sequence[RawPrediction],
    thresholds: GuardThresholds,
) -> dict[str, Any]:
    holdout = [row for row in rows if row.cohort == "holdout"]
    if not holdout:
        raise ReleaseBenchmarkError("benchmark has no holdout rows")
    report: dict[str, Any] = {}
    for dataset in sorted({row.dataset for row in holdout}):
        dataset_rows = [row for row in holdout if row.dataset == dataset]
        by_context: dict[str, Any] = {}
        for variant in sorted({row.context_variant for row in dataset_rows}):
            variant_rows = [row for row in dataset_rows if row.context_variant == variant]
            accepted = [row for row in variant_rows if guard_rejection(row, thresholds) is None]
            hypothesis = [
                row.text if guard_rejection(row, thresholds) is None else "" for row in variant_rows
            ]
            summary: dict[str, Any] = {
                "items": len(variant_rows),
                "accepted": len(accepted),
                "rejected": len(variant_rows) - len(accepted),
                "latency_ms": _latency_summary(variant_rows),
                "decode_limit_rows": sum(row.hit_max_new_tokens is True for row in variant_rows),
            }
            if any(row.reference for row in variant_rows):
                rates = error_rates([row.reference for row in variant_rows], hypothesis)
                summary["quality"] = asdict(rates)
                if dataset == "private_codeswitch":
                    summary["code_switch"] = asdict(
                        code_switch_metrics(
                            [row.reference for row in variant_rows],
                            hypothesis,
                            [row.english_reference_indices for row in variant_rows],
                        )
                    )
            else:
                false_accepts = sum(bool(row.text.strip()) for row in accepted)
                interval = wilson_interval(false_accepts, len(variant_rows))
                summary["hard_negative"] = {
                    "false_accepts": false_accepts,
                    "false_accept_rate": false_accepts / len(variant_rows),
                    "wilson_95": list(interval),
                }
            by_context[variant] = summary
        report[dataset] = by_context

    calibration_rows = [row for row in rows if row.cohort == "calibration"]
    ranked = [
        row for row in calibration_rows if row.avg_logprob is not None and row.avg_logprob_reliable
    ]
    if ranked and {row.dataset == "fleurs_vi" for row in ranked} == {False, True}:
        ranking = ranking_metrics(
            [row.dataset == "fleurs_vi" for row in ranked],
            [float(row.avg_logprob) for row in ranked if row.avg_logprob is not None],
        )
        report["confidence_calibration"] = asdict(ranking)
    report["thresholds"] = asdict(thresholds)
    return report


def context_echo_screening(
    rows: Sequence[RawPrediction],
    *,
    thresholds: Sequence[float],
    minimum_unique_tokens: int,
    contiguous_thresholds: Sequence[int] = (3, 4, 5),
) -> dict[str, Any]:
    contextual = [
        row
        for row in rows
        if row.context_variant == "production_catalog" and row.vad_has_speech and row.text.strip()
    ]
    exact_echoes = sum(row.exact_context_echo for row in contextual)
    sweep: list[dict[str, Any]] = []
    for threshold in thresholds:
        flagged = [
            row
            for row in contextual
            if len(set(normalize(row.text).split())) >= minimum_unique_tokens
            and row.context_unique_token_overlap >= threshold
        ]
        true_positive = sum(row.exact_context_echo for row in flagged)
        false_positive = len(flagged) - true_positive
        sweep.append(
            {
                "threshold": threshold,
                "flagged": len(flagged),
                "exact_echo_true_positive": true_positive,
                "automatic_screen_false_positive": false_positive,
            }
        )
    contiguous_sweep = []
    for threshold in contiguous_thresholds:
        flagged = [row for row in contextual if row.context_max_contiguous_tokens >= threshold]
        contiguous_sweep.append(
            {
                "threshold": threshold,
                "flagged": len(flagged),
                "exact_context_echoes": sum(row.exact_context_echo for row in flagged),
            }
        )
    return {
        "evidence_class": "automatic_screening_requires_manual_review",
        "contextual_nonempty_rows": len(contextual),
        "exact_context_echoes": exact_echoes,
        "threshold_sweep": sweep,
        "contiguous_span_sweep": contiguous_sweep,
    }


def evaluate_context_echo_labels(
    rows: Sequence[RawPrediction],
    labels: Sequence[dict[str, Any]],
    *,
    minimum_contiguous_tokens: int,
) -> dict[str, Any]:
    contextual = {
        (row.dataset, row.item_id): row
        for row in rows
        if row.context_variant == "production_catalog"
    }
    reviewed: set[tuple[str, str]] = set()
    true_positive = false_positive = false_negative = true_negative = 0
    for label in labels:
        if set(label) != {"dataset", "item_id", "context_echo"}:
            raise ReleaseBenchmarkError("context echo label fields do not match schema")
        dataset = label["dataset"]
        item_id = label["item_id"]
        actual = label["context_echo"]
        if (
            not isinstance(dataset, str)
            or not isinstance(item_id, str)
            or not isinstance(actual, bool)
        ):
            raise ReleaseBenchmarkError("context echo label values are invalid")
        key = (dataset, item_id)
        if key in reviewed:
            raise ReleaseBenchmarkError(f"duplicate context echo label: {key}")
        reviewed.add(key)
        row = contextual.get(key)
        if row is None:
            raise ReleaseBenchmarkError(f"context echo label has no prediction: {key}")
        predicted = row.context_max_contiguous_tokens >= minimum_contiguous_tokens
        if predicted and actual:
            true_positive += 1
        elif predicted:
            false_positive += 1
        elif actual:
            false_negative += 1
        else:
            true_negative += 1
    positives = true_positive + false_negative
    negatives = true_negative + false_positive
    return {
        "evidence_class": "manual_paired_decode_review",
        "reviewed": len(reviewed),
        "minimum_contiguous_tokens": minimum_contiguous_tokens,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "context_echo_false_accept_rate": false_negative / positives if positives else 0.0,
        "context_echo_false_reject_rate": false_positive / negatives if negatives else 0.0,
    }


def evaluate_release_quality_gates(
    summaries: dict[str, Any],
    *,
    release_key: str,
    control_key: str,
    gates: dict[str, float | int],
) -> dict[str, Any]:
    release = summaries[release_key]
    control = summaries[control_key]
    release_private = release["private_codeswitch"]["production_catalog"]
    control_private = control["private_codeswitch"]["empty"]
    release_public = release["fleurs_vi"]["empty"]
    control_public = control["fleurs_vi"]["empty"]
    release_noise = release["non_speech"]["production_catalog"]

    checks = [
        _minimum_gate(
            "private_codeswitch_absolute_cs_wer_improvement",
            control_private["code_switch"]["cs_wer"] - release_private["code_switch"]["cs_wer"],
            float(gates["private_codeswitch_min_absolute_cs_wer_improvement"]),
        ),
        _maximum_gate(
            "private_codeswitch_absolute_wer_regression",
            release_private["quality"]["wer"] - control_private["quality"]["wer"],
            float(gates["private_codeswitch_max_absolute_wer_regression"]),
        ),
        _maximum_gate(
            "public_vi_absolute_wer_regression",
            release_public["quality"]["wer"] - control_public["quality"]["wer"],
            float(gates["public_vi_max_absolute_wer_regression"]),
        ),
        _maximum_gate(
            "public_vi_speech_false_reject_rate",
            release_public["rejected"] / release_public["items"],
            float(gates["speech_false_reject_rate_max"]),
        ),
        _maximum_gate(
            "hard_negative_false_accept_rate",
            release_noise["hard_negative"]["false_accept_rate"],
            float(gates["hard_negative_false_accept_rate_max"]),
        ),
        _maximum_gate(
            "final_rtf_p95",
            release_private["latency_ms"]["rtf_p95"],
            float(gates["final_rtf_p95_max"]),
        ),
    ]
    manual_echo = release.get("context_echo_manual_review")
    if isinstance(manual_echo, dict):
        checks.extend(
            [
                _maximum_gate(
                    "context_echo_false_accept_rate",
                    manual_echo.get("context_echo_false_accept_rate"),
                    float(gates["context_echo_false_accept_rate_max"]),
                ),
                _maximum_gate(
                    "context_echo_false_reject_rate",
                    manual_echo.get("context_echo_false_reject_rate"),
                    float(gates["context_echo_false_reject_rate_max"]),
                ),
            ]
        )
    failures = [check["name"] for check in checks if not check["passed"]]
    pending = [] if isinstance(manual_echo, dict) else ["manual_context_echo_labels"]
    pending.extend(
        [
            "partial_stability_and_latency",
            "cold_start_and_resource_repetitions",
            "real_failure_and_start_stop_stress",
            "full_voice_remote_llm_tts_trajectory",
        ]
    )
    return {
        "status": "failed_quality_gate" if failures else "incomplete",
        "checks": checks,
        "failed": failures,
        "pending": pending,
    }


def _minimum_gate(name: str, value: float, minimum: float) -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "operator": ">=",
        "limit": minimum,
        "passed": value >= minimum,
    }


def _maximum_gate(name: str, value: float | None, maximum: float) -> dict[str, Any]:
    passed = value is not None and math.isfinite(value) and value <= maximum
    return {
        "name": name,
        "value": value,
        "operator": "<=",
        "limit": maximum,
        "passed": passed,
    }


def calibration_record(
    *,
    identity: ASRCalibrationIdentity,
    thresholds: GuardThresholds,
    config: ASRReleaseConfig,
    dataset_digests: dict[str, str],
    rows: Sequence[RawPrediction],
    run_id: str,
) -> dict[str, Any]:
    calibration = [row for row in rows if row.cohort == "calibration"]
    return {
        "identity": identity.payload,
        "recommended_thresholds": asdict(thresholds),
        "selection": {
            "algorithm": "speech_lower_bound_and_compression_p99_v1",
            **config.threshold_selection,
        },
        "dataset_digests": dataset_digests,
        "calibration_rows": {
            "speech": sum(row.dataset == "fleurs_vi" for row in calibration),
            "non_speech": sum(row.dataset == "non_speech" for row in calibration),
        },
        "benchmark_config_digest": config.digest,
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }


def write_raw_run(
    output_dir: Path,
    *,
    metadata: dict[str, Any],
    rows_by_artifact: dict[str, Sequence[RawPrediction]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "run.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for artifact_key, rows in rows_by_artifact.items():
        path = output_dir / f"{artifact_key}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def new_run_id() -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


def runtime_metadata(config: ASRReleaseConfig, dataset_digests: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_type": "release_benchmark",
        "run_name": config.run_name,
        "benchmark_config": str(config.source_path),
        "benchmark_config_digest": config.digest,
        "datasets": dataset_digests,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": platform.python_version(),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "fallback_attempted": False,
    }


def open_qwen_backend(
    artifact_key: str,
    *,
    detector: SpeechDetector,
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
) -> tuple[QwenASRServiceClient, ASRCalibrationIdentity]:
    spec = get_qwen_artifact(artifact_key)
    receipt = QwenArtifactStore(default_asr_model_root()).verify(spec, deep=False)
    identity = qwen_calibration_identity(
        spec,
        vad_policy_digest=compute_vad_policy_digest(detector),
    )
    client = QwenASRServiceClient(
        launch=QwenServiceLaunch.for_active(spec, receipt),
        request_timeout_s=request_timeout_s,
    )
    if client.identity is None or not client.identity.no_fallback_attempted:
        client.close()
        raise ReleaseBenchmarkError("Qwen service did not attest the no-fallback policy")
    return client, identity


def write_calibration_artifact(
    path: Path,
    records: dict[str, dict[str, Any]],
) -> None:
    if not records or any(len(key) != 64 for key in records):
        raise ReleaseBenchmarkError("calibration records require SHA-256 identity keys")
    payload = {
        "schema_version": 1,
        "calibrations": {key: records[key] for key in sorted(records)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_audio(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    result = np.asarray(audio, dtype=np.float32)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ReleaseBenchmarkError(f"audio is not finite mono PCM: {path}")
    return result


def _latency_summary(rows: Sequence[RawPrediction]) -> dict[str, float | None]:
    latencies = [row.asr_latency_ms for row in rows if row.vad_has_speech]
    rtfs = [row.rtf for row in rows if row.vad_has_speech]
    if not latencies:
        return {"p50_ms": None, "p95_ms": None, "p99_ms": None, "rtf_p95": None}
    return {
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "p99_ms": percentile(latencies, 0.99),
        "rtf_p95": percentile(rtfs, 0.95),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ReleaseBenchmarkError(f"JSONL row is not an object: {path}:{line_number}")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseBenchmarkError(f"cannot read benchmark manifest: {path}") from exc
    if not rows:
        raise ReleaseBenchmarkError(f"benchmark manifest is empty: {path}")
    return rows


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ReleaseBenchmarkError(f"manifest field {key} must be a non-empty string")
    return value


def _context_token_overlap(text: str, context: str) -> float:
    hypothesis_tokens = set(normalize(text).split())
    if not hypothesis_tokens or not context.strip():
        return 0.0
    context_tokens = set(normalize(context).split())
    return len(hypothesis_tokens & context_tokens) / len(hypothesis_tokens)


def _exact_context_echo(text: str, context: str) -> bool:
    hypothesis = normalize(text)
    normalized_context = normalize(context)
    return bool(
        hypothesis
        and normalized_context
        and len(set(hypothesis.split())) >= 4
        and (hypothesis == normalized_context or hypothesis in normalized_context)
    )


def _resolve_audio_path(root: Path, manifest: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    root_candidate = root / candidate
    if root_candidate.is_file():
        return root_candidate
    return manifest.parent / candidate


__all__ = [
    "BenchmarkItem",
    "GuardThresholds",
    "RawPrediction",
    "ReleaseBenchmarkError",
    "build_context_factory",
    "calibration_record",
    "collect_predictions",
    "context_echo_screening",
    "derive_thresholds",
    "evaluate_release_quality_gates",
    "evaluate_context_echo_labels",
    "guard_rejection",
    "load_release_items",
    "new_run_id",
    "open_qwen_backend",
    "prediction_key",
    "runtime_metadata",
    "summarize_predictions",
    "write_raw_run",
    "write_calibration_artifact",
]
