"""Benchmark SoCa working-memory summaries with annotated state facts.

This evaluator intentionally keeps SoCa's synthetic state-retention suite
separate from public document/dialogue summarization. Exact-string metrics are
retained only as diagnostics; release decisions use annotated anchor coverage,
schema validity, stale/unsafe surface leakage, latency, and process telemetry.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from soca.llm import LocalLlamaCppLLM
from soca.memory.summary import (
    SUMMARY_MODEL_REGISTRY,
    LocalSummaryWorkerProcess,
    SummaryModelSpec,
    execute_summary_job,
)
from soca.memory.working import CompactionJob, ConversationTurn, WorkingSummaryArtifact

_STRUCTURED_FIELDS = (
    "user_constraints",
    "decisions",
    "corrections",
    "open_items",
    "continuity_refs",
)


def _artifact(payload: dict[str, Any]) -> WorkingSummaryArtifact:
    return WorkingSummaryArtifact(
        version=int(payload["version"]),
        generation=int(payload["generation"]),
        source_through_sequence=int(payload["source_through_sequence"]),
        summary=str(payload["summary"]),
        user_constraints=tuple(payload.get("user_constraints", ())),
        decisions=tuple(payload.get("decisions", ())),
        corrections=tuple(payload.get("corrections", ())),
        open_items=tuple(payload.get("open_items", ())),
        continuity_refs=tuple(payload.get("continuity_refs", ())),
        prompt_fingerprint=str(payload.get("prompt_fingerprint", "")),
    )


def _turns(values: list[dict[str, Any]]) -> tuple[ConversationTurn, ...]:
    return tuple(
        ConversationTurn(
            sequence=int(item["sequence"]),
            user_text=str(item["user"]),
            assistant_text=str(item["assistant"]),
            status="complete",
        )
        for item in values
    )


def _job(
    row: dict[str, Any],
    generation: int,
    *,
    previous: WorkingSummaryArtifact | None = None,
) -> CompactionJob:
    if previous is None and row.get("previous_summary"):
        previous_payload = dict(row["previous_summary"])
        previous_payload.setdefault("version", 1)
        previous_payload.setdefault("generation", max(0, generation - 1))
        previous_payload.setdefault("source_through_sequence", 0)
        previous = _artifact(previous_payload)
    return CompactionJob(
        generation=generation,
        revision=generation,
        previous_summary=previous,
        frozen_turns=_turns(row["frozen_turns"]),
    )


def _normalise(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _field_recall(expected: dict[str, Any], actual: WorkingSummaryArtifact) -> float:
    """Legacy exact-string diagnostic; do not use as the release quality gate."""
    wanted: set[str] = set()
    got: set[str] = set()
    for field in _STRUCTURED_FIELDS:
        wanted.update(str(value) for value in expected.get(field, []))
        got.update(getattr(actual, field))
    if not wanted:
        return 1.0 if not got else 0.0
    return len(wanted & got) / len(wanted)


def _concept_assessment(
    required_facts: list[dict[str, Any]],
    actual: WorkingSummaryArtifact,
) -> dict[str, object]:
    """Score human-authored fact anchors inside their expected structured field."""
    by_field: dict[str, list[bool]] = defaultdict(list)
    missed: list[dict[str, object]] = []
    for fact in required_facts:
        field = str(fact["field"])
        if field not in _STRUCTURED_FIELDS:
            raise ValueError(f"unknown required-fact field: {field}")
        anchors = tuple(_normalise(str(value)) for value in fact.get("anchors", ()))
        rendered = _normalise("\n".join(getattr(actual, field)))
        matched = bool(anchors) and all(anchor in rendered for anchor in anchors)
        by_field[field].append(matched)
        if not matched:
            missed.append({"field": field, "anchors": list(fact.get("anchors", ()))})
    scores = {
        field: sum(matches) / len(matches)
        for field, matches in sorted(by_field.items())
        if matches
    }
    all_matches = [match for matches in by_field.values() for match in matches]
    return {
        "required_fact_recall": sum(all_matches) / len(all_matches) if all_matches else 1.0,
        "required_fact_recall_by_field": scores,
        "missed_required_facts": missed,
    }


def _assess_artifact(
    expected: dict[str, Any],
    actual: WorkingSummaryArtifact,
    *,
    required_facts: list[dict[str, Any]] | None = None,
    forbidden_claims: list[str],
) -> dict[str, object]:
    """Score annotated state without pretending prose must exactly match a reference."""
    exact_field_recall: dict[str, float] = {}
    unexpected: set[str] = set()
    for field in _STRUCTURED_FIELDS:
        wanted = {str(value) for value in expected.get(field, [])}
        got = set(getattr(actual, field))
        exact_field_recall[field] = len(wanted & got) / len(wanted) if wanted else float(not got)
        unexpected.update(got - wanted)
    rendered = _normalise(
        "\n".join(
            [
                actual.summary,
                *actual.user_constraints,
                *actual.decisions,
                *actual.corrections,
                *actual.open_items,
                *actual.continuity_refs,
            ]
        )
    )
    forbidden = [claim for claim in forbidden_claims if _normalise(claim) in rendered]
    expected_is_empty = not str(expected.get("summary", "")).strip() and not any(
        expected.get(field, []) for field in _STRUCTURED_FIELDS
    )
    structured_state_is_empty = not any(getattr(actual, field) for field in _STRUCTURED_FIELDS)
    return {
        "exact_field_recall_by_field": exact_field_recall,
        "unexpected_items": sorted(unexpected),
        "forbidden_surface_matches": forbidden,
        "negative_state_case": expected_is_empty,
        "negative_state_clean": structured_state_is_empty if expected_is_empty else None,
        **_concept_assessment(required_facts or [], actual),
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def _cold_execute(
    job: CompactionJob,
    spec: SummaryModelSpec,
    *,
    model_root: Path,
    timeout_seconds: float,
    threads: int,
    gpu_layers: int,
) -> tuple[WorkingSummaryArtifact, dict[str, Any]]:
    worker = LocalSummaryWorkerProcess(
        spec,
        model_root=model_root,
        n_threads=threads,
        n_gpu_layers=gpu_layers,
    )
    if not worker.start(job):
        raise RuntimeError("summary worker did not start")
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        payload = worker.poll()
        if payload is None:
            time.sleep(0.01)
            continue
        if not payload.get("ok"):
            raise RuntimeError(str(payload.get("error", "summary worker failed")))
        artifact_payload = payload.get("artifact")
        if not isinstance(artifact_payload, dict):
            raise RuntimeError("summary worker returned no artifact")
        return _artifact(artifact_payload), payload
    worker.cancel()
    raise TimeoutError("summary worker exceeded timeout")


def _evaluate_row(
    row: dict[str, Any],
    *,
    execute: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    telemetry: list[dict[str, Any]] = []
    artifacts: list[WorkingSummaryArtifact] = []
    previous: WorkingSummaryArtifact | None = None
    if "generations" in row:
        generations = list(row["generations"])
        for generation, value in enumerate(generations, start=1):
            job = _job(value, generation, previous=previous)
            previous, payload = execute(job)
            artifacts.append(previous)
            telemetry.append(payload)
        expected = row["expected_final"]
    else:
        artifact, payload = execute(_job(row, 1))
        artifacts.append(artifact)
        telemetry.append(payload)
        expected = row["expected"]
    actual = artifacts[-1]
    assessment = _assess_artifact(
        expected,
        actual,
        required_facts=list(row.get("required_facts", [])),
        forbidden_claims=list(row.get("forbidden_claims", [])),
    )
    return {
        "id": row["id"],
        "family": row["family"],
        "ok": True,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "schema_valid": True,
        "legacy_exact_field_recall": _field_recall(expected, actual),
        **assessment,
        "summary_tokens": len(actual.summary.split()),
        "artifacts": [value.to_dict() for value in artifacts],
        "worker_telemetry": telemetry,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if args.per_family_limit:
        selected: list[dict[str, Any]] = []
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            family = str(row["family"])
            if counts[family] >= args.per_family_limit:
                continue
            selected.append(row)
            counts[family] += 1
        rows = selected
    if args.limit:
        rows = rows[: args.limit]
    try:
        candidate = SUMMARY_MODEL_REGISTRY[args.model_key]
    except KeyError as exc:
        valid = ", ".join(sorted(SUMMARY_MODEL_REGISTRY))
        raise ValueError(f"unknown summary candidate: {args.model_key}; valid: {valid}") from exc

    engine: LocalLlamaCppLLM | None = None
    model_root = args.model_path.parents[2]
    if args.cold_process:
        expected_path = candidate.path(model_root).resolve()
        if expected_path != args.model_path.resolve():
            raise ValueError(f"model path must match registry layout: {expected_path}")
    else:
        engine = LocalLlamaCppLLM(
            model_key=candidate.key,
            model_path=args.model_path,
            model_config=candidate.runtime_config(),
            n_ctx=4096,
            n_threads=args.threads,
            n_gpu_layers=args.gpu_layers,
        )

    def execute(job: CompactionJob) -> tuple[WorkingSummaryArtifact, dict[str, Any]]:
        if args.cold_process:
            return _cold_execute(
                job,
                candidate,
                model_root=model_root,
                timeout_seconds=args.timeout_seconds,
                threads=args.threads,
                gpu_layers=args.gpu_layers,
            )
        assert engine is not None
        artifact, usage = execute_summary_job(job, engine)
        return artifact, {"usage": usage.to_dict()}

    records: list[dict[str, Any]] = []
    for row in rows:
        try:
            records.append(_evaluate_row(row, execute=execute))
        except Exception as exc:  # noqa: BLE001 - bake-off captures per-case failures
            records.append(
                {
                    "id": row["id"],
                    "family": row["family"],
                    "ok": False,
                    "schema_valid": False,
                    "error": type(exc).__name__,
                    "error_detail": str(exc)[:240],
                }
            )

    successful = [record for record in records if record.get("ok")]
    latencies = [float(record["latency_ms"]) for record in successful]
    fact_scores = [
        float(record["required_fact_recall"])
        for record in successful
        if not record["negative_state_case"]
    ]
    negative_records = [record for record in successful if record["negative_state_case"]]
    by_family: dict[str, dict[str, float | int | None]] = {}
    for family in sorted({str(record["family"]) for record in records}):
        family_records = [record for record in records if record["family"] == family]
        family_ok = [record for record in family_records if record.get("ok")]
        positive = [record for record in family_ok if not record["negative_state_case"]]
        negative = [record for record in family_ok if record["negative_state_case"]]
        by_family[family] = {
            "records": len(family_records),
            "schema_valid_rate": len(family_ok) / len(family_records),
            "required_fact_recall": (
                sum(float(record["required_fact_recall"]) for record in positive) / len(positive)
                if positive
                else None
            ),
            "negative_state_clean_rate": (
                sum(bool(record["negative_state_clean"]) for record in negative) / len(negative)
                if negative
                else None
            ),
            "forbidden_surface_match_rate": (
                sum(bool(record["forbidden_surface_matches"]) for record in family_ok) / len(family_ok)
                if family_ok
                else 0.0
            ),
        }
    load_times = [
        float(value["load_latency_ms"])
        for record in successful
        for value in record["worker_telemetry"]
        if value.get("load_latency_ms") is not None
    ]
    generation_times = [
        float(value["generation_latency_ms"])
        for record in successful
        for value in record["worker_telemetry"]
        if value.get("generation_latency_ms") is not None
    ]
    peak_rss = [
        float(value["peak_rss_mb"])
        for record in successful
        for value in record["worker_telemetry"]
        if value.get("peak_rss_mb") is not None
    ]
    context_windows = [
        int(value["n_ctx"])
        for record in successful
        for value in record["worker_telemetry"]
        if value.get("n_ctx") is not None
    ]
    cold_payloads = [
        value
        for record in successful
        for value in record["worker_telemetry"]
        if value.get("load_latency_ms") is not None
    ]
    dataset_version = str(rows[0].get("dataset_version", "unknown")) if rows else "unknown"
    return {
        "benchmark": dataset_version,
        "dataset": str(args.dataset),
        "dataset_rows": len(rows),
        "cold_process_per_job": args.cold_process,
        "mode": "rolling" if rows and "generations" in rows[0] else "single_generation",
        "candidate": {"model_key": args.model_key, "model_path": str(args.model_path)},
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "threads": args.threads,
            "gpu_layers": args.gpu_layers,
            "n_ctx": 4096 if not args.cold_process else None,
            "n_ctx_mode": "fixed" if not args.cold_process else "dynamic",
            "n_ctx_max": candidate.context_window,
            "temperature": 0,
            "max_tokens": 384,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        },
        "metrics": {
            "schema_valid_rate": len(successful) / len(rows) if rows else 0.0,
            "required_fact_recall_mean": (
                sum(fact_scores) / len(fact_scores) if fact_scores else 0.0
            ),
            "negative_state_clean_rate": (
                sum(bool(record["negative_state_clean"]) for record in negative_records)
                / len(negative_records)
                if negative_records
                else None
            ),
            "forbidden_surface_match_record_rate": (
                sum(bool(record["forbidden_surface_matches"]) for record in successful)
                / len(successful)
                if successful
                else 0.0
            ),
            "unexpected_item_record_rate_diagnostic": (
                sum(bool(record["unexpected_items"]) for record in successful) / len(successful)
                if successful
                else 0.0
            ),
            "latency_ms_p50": _percentile(latencies, 0.50),
            "latency_ms_p95": _percentile(latencies, 0.95),
            "cold_load_latency_ms_p50": _percentile(load_times, 0.50),
            "cold_generation_latency_ms_p50": _percentile(generation_times, 0.50),
            "cold_peak_rss_mb_max": max(peak_rss) if peak_rss else None,
            "cold_n_ctx_max": max(context_windows) if context_windows else None,
            "cold_clean_exit_rate": (
                sum(value.get("exit_code") == 0 for value in cold_payloads) / len(cold_payloads)
                if cold_payloads
                else None
            ),
            "cold_worker_stopped_rate": (
                sum(bool(value.get("worker_stopped")) for value in cold_payloads) / len(cold_payloads)
                if cold_payloads
                else None
            ),
            "by_family": by_family,
        },
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-family-limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--gpu-layers", type=int, default=-1)
    parser.add_argument("--cold-process", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(run(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output.chmod(0o600)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
