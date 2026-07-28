"""Run a reproducible structured-summary candidate capture (explicitly provisioned models only)."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from soca.llm import LocalLlamaCppLLM
from soca.memory.summary import SUMMARY_MODEL_REGISTRY, execute_summary_job
from soca.memory.working import CompactionJob, ConversationTurn, WorkingSummaryArtifact

_STRUCTURED_FIELDS = (
    "user_constraints",
    "decisions",
    "corrections",
    "open_items",
    "continuity_refs",
)


def _job(row: dict[str, Any], generation: int) -> CompactionJob:
    previous = row.get("previous_summary") or None
    artifact = None
    if previous:
        artifact = WorkingSummaryArtifact(
            version=1,
            generation=max(0, generation - 1),
            source_through_sequence=0,
            summary=str(previous.get("summary", "")),
            user_constraints=tuple(previous.get("user_constraints", ())),
            decisions=tuple(previous.get("decisions", ())),
            corrections=tuple(previous.get("corrections", ())),
            open_items=tuple(previous.get("open_items", ())),
            continuity_refs=tuple(previous.get("continuity_refs", ())),
        )
    turns = tuple(
        ConversationTurn(
            sequence=int(item["sequence"]),
            user_text=str(item["user"]),
            assistant_text=str(item["assistant"]),
            status="complete",
        )
        for item in row["frozen_turns"]
    )
    return CompactionJob(generation=generation, revision=generation, previous_summary=artifact, frozen_turns=turns)


def _field_recall(expected: dict[str, Any], actual: WorkingSummaryArtifact) -> float:
    wanted: set[str] = set()
    got: set[str] = set()
    for field in _STRUCTURED_FIELDS:
        wanted.update(str(value) for value in expected.get(field, []))
        got.update(getattr(actual, field))
    if not wanted:
        return 1.0 if not got else 0.0
    return len(wanted & got) / len(wanted)


def _assess_artifact(
    expected: dict[str, Any],
    actual: WorkingSummaryArtifact,
    *,
    forbidden_claims: list[str],
) -> dict[str, object]:
    """Deterministically score only annotated structured state, never prose similarity."""
    field_recall: dict[str, float] = {}
    unexpected: set[str] = set()
    for field in _STRUCTURED_FIELDS:
        wanted = {str(value) for value in expected.get(field, [])}
        got = set(getattr(actual, field))
        field_recall[field] = len(wanted & got) / len(wanted) if wanted else float(not got)
        unexpected.update(got - wanted)
    rendered = "\n".join(
        [
            actual.summary,
            *actual.user_constraints,
            *actual.decisions,
            *actual.corrections,
            *actual.open_items,
            *actual.continuity_refs,
        ]
    ).casefold()
    forbidden = [claim for claim in forbidden_claims if claim.casefold() in rendered]
    return {
        "field_recall_by_field": field_recall,
        "unexpected_items": sorted(unexpected),
        "forbidden_leaks": forbidden,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line]
    if args.limit:
        rows = rows[: args.limit]
    try:
        candidate = SUMMARY_MODEL_REGISTRY[args.model_key]
    except KeyError as exc:
        valid = ", ".join(sorted(SUMMARY_MODEL_REGISTRY))
        raise ValueError(f"unknown summary candidate: {args.model_key}; valid: {valid}") from exc
    engine = LocalLlamaCppLLM(
        model_key=candidate.key,
        model_path=args.model_path,
        model_config=candidate.runtime_config(),
        n_ctx=4096,
        n_threads=args.threads,
        n_gpu_layers=args.gpu_layers,
    )
    records: list[dict[str, Any]] = []
    for generation, row in enumerate(rows, start=1):
        started = time.perf_counter()
        try:
            artifact, usage = execute_summary_job(_job(row, generation), engine)
            assessment = _assess_artifact(
                row["expected"],
                artifact,
                forbidden_claims=list(row.get("forbidden_claims", [])),
            )
            records.append(
                {
                    "id": row["id"],
                    "ok": True,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "schema_valid": True,
                    "field_recall": _field_recall(row["expected"], artifact),
                    **assessment,
                    "summary_tokens": len(artifact.summary.split()),
                    "prompt_tokens": usage.n_prompt_tokens,
                    "completion_tokens": usage.n_completion_tokens,
                }
            )
        except Exception as exc:  # noqa: BLE001 - bake-off must capture failures
            records.append({"id": row["id"], "ok": False, "error": type(exc).__name__})
    latencies = sorted(record["latency_ms"] for record in records if record.get("ok"))
    recall = [record["field_recall"] for record in records if record.get("ok")]
    successful = [record for record in records if record.get("ok")]
    field_metrics = {
        field: sum(record["field_recall_by_field"][field] for record in successful) / len(successful)
        if successful
        else 0.0
        for field in _STRUCTURED_FIELDS
    }
    return {
        "benchmark": "summary_session_vi_v1",
        "dataset": str(args.dataset),
        "dataset_rows": len(rows),
        "cold_process_per_job": False,
        "mode": "single_process_smoke_not_release_gate",
        "candidate": {"model_key": args.model_key, "model_path": str(args.model_path)},
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "threads": args.threads,
            "gpu_layers": args.gpu_layers,
            "n_ctx": 4096,
            "temperature": 0,
            "max_tokens": 384,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        },
        "metrics": {
            "schema_valid_rate": sum(bool(record.get("schema_valid")) for record in records) / len(rows) if rows else 0.0,
            "field_recall_mean": sum(recall) / len(recall) if recall else 0.0,
            "field_recall_by_field": field_metrics,
            "unexpected_item_record_rate": sum(bool(record["unexpected_items"]) for record in successful)
            / len(successful)
            if successful
            else 0.0,
            "forbidden_leak_record_rate": sum(bool(record["forbidden_leaks"]) for record in successful)
            / len(successful)
            if successful
            else 0.0,
            "latency_ms_p50": latencies[len(latencies) // 2] if latencies else None,
            "latency_ms_p95": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else None,
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
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--gpu-layers", type=int, default=-1)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(run(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
