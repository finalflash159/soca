from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from eval.baseline_cases import RemediationCase, load_cases
from eval.remediation_eval import DEFAULT_CORPUS_ROOT, DEFAULT_DATASETS, validate_corpus
from eval.result_io import make_eval_artifact_metadata, write_json_artifact
from soca.app.text_runtime import TextRuntimeConfig, build_text_runtime
from soca.config import LlmSettings, load_settings
from soca.core import RuntimeResult, RuntimeRoute

REPO_ROOT = Path(__file__).resolve().parents[1]
_SECRET_RE = re.compile(r"\b(?:sk|AIza)[A-Za-z0-9_-]{8,}\b")


class BaselineRuntime(Protocol):
    def run_text_turn(
        self,
        text: str,
        *,
        source: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeResult: ...

    def stream_text_turn(
        self,
        text: str,
        *,
        source: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> Iterable[Any]: ...


@dataclass(frozen=True)
class BaselineRunConfig:
    execution_mode: str
    provider: str
    model: str
    backend: str
    retrieval_mode: str
    dense_backend: str
    max_tokens: int


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list | set):
        return [_json_value(item) for item in value]
    return str(value)


def _run_turn(
    runtime: BaselineRuntime,
    text: str,
    *,
    execution_mode: str,
    case_id: str,
    turn_index: int,
) -> RuntimeResult:
    metadata = {"eval_case_id": case_id, "eval_turn_index": turn_index}
    if execution_mode == "blocking":
        return runtime.run_text_turn(text, source="text", metadata=metadata)
    if execution_mode != "streaming":
        raise ValueError(f"unsupported execution mode: {execution_mode}")
    terminal: RuntimeResult | None = None
    for event in runtime.stream_text_turn(text, source="text", metadata=metadata):
        if getattr(event, "type", "") == "result":
            terminal = getattr(event, "result", None)
    if terminal is None:
        raise RuntimeError("stream ended without a terminal RuntimeResult")
    return terminal


def _terminal_from_result(result: RuntimeResult) -> str:
    trace = result.trace
    if result.blocked or result.route is RuntimeRoute.BLOCKED:
        return "safe_failure"
    if result.route is RuntimeRoute.CLARIFICATION:
        return "needs_clarification"
    if result.route is RuntimeRoute.OUT_OF_SCOPE:
        return "safe_failure"
    evidence_status = str(getattr(trace, "evidence_status", "not_requested"))
    selected_sources = tuple(getattr(trace, "selected_sources", ()))
    used_retrieval = bool(
        selected_sources
        or any(
            call.name.startswith(("knowledge.", "memory."))
            for call in getattr(trace, "tool_calls", ())
        )
    )
    if used_retrieval and evidence_status in {"insufficient", "unavailable"}:
        return "insufficient_evidence"
    return "achieved"


def _actual_sources(result: RuntimeResult) -> tuple[str, ...]:
    trace = result.trace
    values = set(getattr(trace, "selected_sources", ()))
    values.update(citation.source for citation in result.citations)
    for call in getattr(trace, "tool_calls", ()):
        prefix = call.name.partition(".")[0]
        if prefix in {"knowledge", "memory"}:
            values.add(prefix)
    return tuple(sorted(values))


def _actual_citations(result: RuntimeResult) -> tuple[str, ...]:
    values = {citation.path for citation in result.citations}
    values.update(citation.source for citation in result.citations)
    return tuple(sorted(values))


def _turn_record(
    result: RuntimeResult,
    *,
    user_text: str,
    elapsed_ms: float,
    turn_index: int,
) -> dict[str, Any]:
    trace = result.trace
    usage = result.usage
    return {
        "turn_index": turn_index,
        "input": user_text,
        "response": result.response_text,
        "route": result.route.value,
        "blocked": result.blocked,
        "terminal": _terminal_from_result(result),
        "tool_calls": _json_value(getattr(trace, "tool_calls", ())),
        "tool_results": _json_value(getattr(trace, "tool_results", ())),
        "selected_sources": list(_actual_sources(result)),
        "selected_routes": list(getattr(trace, "selected_routes", ())),
        "citations": _json_value(result.citations),
        "knowledge_hits": _json_value(getattr(trace, "knowledge_hits", ())),
        "memory_hits": _json_value(getattr(trace, "memory_hits", ())),
        "evidence_status": str(getattr(trace, "evidence_status", "not_requested")),
        "evidence_decisions": _json_value(getattr(trace, "evidence_decisions", ())),
        "evidence_bundle": _json_value(getattr(trace, "evidence_bundle", None)),
        "answer_policy": str(getattr(trace, "answer_policy", "")),
        "answer_policy_reason": str(getattr(trace, "answer_policy_reason", "")),
        "answer_validation": _json_value(getattr(trace, "answer_validation", None)),
        "prompt_manifest": _json_value(getattr(trace, "prompt_manifest", None)),
        "router": {
            "tier": str(getattr(trace, "tool_router_tier", "none")),
            "reason": str(getattr(trace, "tool_router_reason", "no_match")),
            "disposition": str(getattr(trace, "disposition", "unresolved")),
            "handler": getattr(trace, "router_handler", None),
            "scores": _json_value(getattr(trace, "router_scores", {})),
        },
        "latency_ms": {
            "wall": round(elapsed_ms, 3),
            "stages": _json_value(getattr(trace, "stage_latencies_ms", {})),
        },
        "usage": _json_value(usage),
        "legacy_terminal": {
            "status": "failed" if result.blocked else "succeeded",
            "route": result.route.value,
        },
    }


def _check_case(case: RemediationCase, turns: list[dict[str, Any]]) -> dict[str, Any]:
    tool_names = {
        str(call.get("name", ""))
        for turn in turns
        for call in turn["tool_calls"]
        if isinstance(call, dict)
    }
    sources = {
        source
        for turn in turns
        for source in turn["selected_sources"]
    }
    citations = {
        value
        for turn in turns
        for value in (
            item
            for citation in turn["citations"]
            if isinstance(citation, dict)
            for item in (citation.get("source"), citation.get("path"))
            if isinstance(item, str)
        )
    }
    actual_terminal = turns[-1]["terminal"] if turns else "system_failure"
    checks = {
        "terminal": actual_terminal == case.expected_terminal,
        "tools": set(case.expected_tools).issubset(tool_names),
        "sources": set(case.expected_sources).issubset(sources),
        "citations": set(case.expected_citations).issubset(citations),
    }
    return {
        "expected": {
            "goal": case.expected_goal,
            "terminal": case.expected_terminal,
            "tools": list(case.expected_tools),
            "sources": list(case.expected_sources),
            "citations": list(case.expected_citations),
        },
        "actual": {
            "terminal": actual_terminal,
            "tools": sorted(tool_names),
            "sources": sorted(sources),
            "citations": sorted(citations),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def run_runtime_baseline(
    cases: Sequence[RemediationCase],
    *,
    runtime: BaselineRuntime,
    reset_case: Callable[[], None] | None = None,
    execution_mode: str = "blocking",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in cases:
        if reset_case is not None:
            reset_case()
        turns: list[dict[str, Any]] = []
        error: dict[str, Any] | None = None
        for turn_index, text in enumerate(case.turns, start=1):
            started = time.perf_counter()
            try:
                result = _run_turn(
                    runtime,
                    text,
                    execution_mode=execution_mode,
                    case_id=case.case_id,
                    turn_index=turn_index,
                )
            except Exception as exc:  # noqa: BLE001 - baseline must record terminal failures
                error = {
                    "turn_index": turn_index,
                    "type": type(exc).__name__,
                    "category": str(getattr(exc, "category", "")),
                    "message": _SECRET_RE.sub("[REDACTED]", str(exc))[:1000],
                }
                break
            turns.append(
                _turn_record(
                    result,
                    user_text=text,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    turn_index=turn_index,
                )
            )
        outcome = (
            _check_case(case, turns)
            if error is None
            else {
                "expected": {
                    "goal": case.expected_goal,
                    "terminal": case.expected_terminal,
                    "tools": list(case.expected_tools),
                    "sources": list(case.expected_sources),
                    "citations": list(case.expected_citations),
                },
                "actual": {
                    "terminal": "system_failure",
                    "tools": [],
                    "sources": [],
                    "citations": [],
                },
                "checks": {
                    "terminal": case.expected_terminal == "system_failure",
                    "tools": not case.expected_tools,
                    "sources": not case.expected_sources,
                    "citations": not case.expected_citations,
                },
                "passed": False,
            }
        )
        records.append(
            {
                "case_id": case.case_id,
                "suite_kind": case.suite_kind,
                "dataset_class": case.dataset_class,
                "split": case.split,
                "family": case.family,
                "category": case.category,
                "audit_items": list(case.audit_items),
                "turns": turns,
                "error": error,
                "outcome": outcome,
            }
        )
    return records


def _summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    suites: dict[str, dict[str, int]] = {}
    for suite_kind in sorted({str(record["suite_kind"]) for record in records}):
        selected = [record for record in records if record["suite_kind"] == suite_kind]
        passed = sum(bool(record["outcome"]["passed"]) for record in selected)
        suites[suite_kind] = {
            "cases": len(selected),
            "passed": passed,
            "failed": len(selected) - passed,
        }
    passed = sum(bool(record["outcome"]["passed"]) for record in records)
    return {
        "cases": len(records),
        "passed": passed,
        "failed": len(records) - passed,
        "by_suite": suites,
        "terminal_counts": dict(
            sorted(Counter(record["outcome"]["actual"]["terminal"] for record in records).items())
        ),
        "error_count": sum(record["error"] is not None for record in records),
    }


def build_runtime_report(
    *,
    records: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    run_config: BaselineRunConfig,
) -> dict[str, Any]:
    summary = _summary(records)
    return {
        "schema_version": "soca-runtime-remediation-baseline-v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "artifact": metadata,
        "run_config": _json_value(run_config),
        "decision": {
            "status": "record_only",
            "reason": "runtime_characterization_baseline",
            "passed": summary["passed"],
            "failed": summary["failed"],
            "error_count": summary["error_count"],
        },
        "summary": summary,
        "cases": list(records),
    }


def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the runtime remediation baseline.")
    parser.add_argument("--dataset", action="append", type=Path, dest="datasets")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--suite-kind", choices=("regression", "capability", "all"), default="all")
    parser.add_argument("--execution-mode", choices=("blocking", "streaming"), default="blocking")
    parser.add_argument("--retrieval-mode", default="hybrid")
    parser.add_argument("--dense-backend", default="aiteamvn_v2")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--ignore-source-path", action="append", type=Path, default=[])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "eval" / "results" / "remediation_baseline" / "current",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_paths = tuple(args.datasets or DEFAULT_DATASETS)
    cases = tuple(case for path in dataset_paths for case in load_cases(path, quality_suite=True))
    if args.suite_kind != "all":
        cases = tuple(case for case in cases if case.suite_kind == args.suite_kind)
    corpus_files = validate_corpus(args.corpus)
    settings = load_settings()
    effective_settings = settings if not args.no_llm else LlmSettings()
    runtime_config = TextRuntimeConfig(
        vault=args.corpus.expanduser().resolve(),
        no_llm=args.no_llm,
        max_tokens=effective_settings.effective_max_tokens,
        temperature=effective_settings.temperature,
        top_p=effective_settings.top_p,
        knowledge_retrieval_mode=args.retrieval_mode,
        knowledge_dense_backend=args.dense_backend,
        session_turns=64,
        turn_chars=4_000,
        session_persistence="ram_only",
        session_id="runtime-remediation-baseline",
    )
    bundle = build_text_runtime(
        runtime_config,
        llm_settings=settings,
    )
    run_config = BaselineRunConfig(
        execution_mode=args.execution_mode,
        provider=settings.provider_key if not args.no_llm else "none",
        model=settings.model_id if not args.no_llm else "none",
        backend=settings.backend if not args.no_llm else "none",
        retrieval_mode=args.retrieval_mode,
        dense_backend=args.dense_backend,
        max_tokens=runtime_config.max_tokens,
    )
    artifact = make_eval_artifact_metadata(
        suite="runtime_remediation_baseline",
        run_type="benchmark",
        data_files=dataset_paths + corpus_files,
        config=_json_value(run_config),
        ignored_untracked_paths=tuple(args.ignore_source_path),
    )
    records = run_runtime_baseline(
        cases,
        runtime=bundle.runtime,
        reset_case=bundle.session_memory.clear if bundle.session_memory is not None else None,
        execution_mode=args.execution_mode,
    )
    report = build_runtime_report(
        records=records,
        metadata=artifact.to_dict(),
        run_config=run_config,
    )
    output_dir = args.output_dir
    write_json_artifact(output_dir / "report.json", report)
    _write_jsonl(output_dir / "run.log.jsonl", records)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if report["summary"]["error_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
