"""Deterministic eval harness for the SoCa AssistantRuntime.

This evaluator intentionally scores runtime traces, not surface text only. It
checks route, tool calls, guardrail reasons, LLM usage, knowledge citations, and
latency stages so regressions in routing or safety are visible before a real
model is loaded.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any
from zoneinfo import ZoneInfo

from rich.console import Console
from rich.progress import track
from rich.table import Table

from soca.core import AssistantRuntime, RuntimeOptions, RuntimeRoute
from soca.knowledge import KnowledgeContextBuilder, KnowledgeDocument, KnowledgeHit
from soca.llm import LLMResult
from soca.memory import MemoryContextBuilder, SessionMemory
from soca.tools import (
    KnowledgeReadTool,
    KnowledgeSearchTool,
    LocalTimeTool,
    ToolResult,
    ToolRuntime,
)
from soca.tools.base import SideEffectLevel, ToolSpec, object_schema

console = Console()
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_PATH = REPO_ROOT / "eval" / "prompts" / "assistant_runtime_vi.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "eval" / "results"
FIXED_NOW = datetime(2026, 5, 29, 9, 30, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))


@dataclass(frozen=True)
class RuntimeEvalCase:
    case_id: str
    category: str
    text: str
    expected_route: RuntimeRoute
    should_block: bool
    scenario: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)
    expected_tool: str | None = None
    requires_tool: bool | None = None
    requires_llm: bool | None = None
    requires_citation: bool = False
    expected_guardrail_reason: str | None = None
    expected_citation_paths: tuple[str, ...] = ()
    expected_knowledge_hit_paths: tuple[str, ...] = ()
    expect_no_knowledge_hits: bool = False
    expected_response_contains: tuple[str, ...] = ()
    fake_llm_response: str = "Mình là SoCa, trợ lý tiếng Việt."


@dataclass(frozen=True)
class RuntimeEvalSample:
    case: RuntimeEvalCase
    response_text: str
    route: str
    blocked: bool
    tool_calls: tuple[str, ...]
    tool_results: tuple[dict[str, Any], ...]
    guardrail_reasons: tuple[str, ...]
    citation_paths: tuple[str, ...]
    knowledge_hit_paths: tuple[str, ...]
    used_tool: bool
    used_llm: bool
    stage_latencies_ms: dict[str, float]
    checks: dict[str, bool]

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


class EvalKnowledgeSource:
    def __init__(self, *, empty: bool = False) -> None:
        self.empty = empty
        self.documents = {
            "wiki/dinh-duong/chat-dam.md": KnowledgeDocument(
                id="chat-dam",
                path="wiki/dinh-duong/chat-dam.md",
                title="Chất đạm",
                text=(
                    "# Chất đạm\n"
                    "Protein hỗ trợ duy trì cơ bắp và cảm giác no. "
                    "Nguồn dễ dùng gồm trứng, cá, đậu phụ, sữa chua và thịt nạc."
                ),
                tags=("dinh-duong", "protein"),
            ),
            "wiki/dinh-duong/bua-sang.md": KnowledgeDocument(
                id="bua-sang",
                path="wiki/dinh-duong/bua-sang.md",
                title="Bữa sáng cân bằng",
                text=(
                    "# Bữa sáng cân bằng\n"
                    "Một bữa sáng tốt nên có chất đạm, tinh bột chậm và rau hoặc trái cây."
                ),
                tags=("dinh-duong", "bua-sang"),
            ),
        }

    def search(self, query: str, limit: int = 5) -> list[KnowledgeHit]:
        if self.empty:
            return []

        normalized = query.lower()
        hits: list[KnowledgeHit] = []
        for doc in self.documents.values():
            haystack = f"{doc.title}\n{doc.text}\n{' '.join(doc.tags)}".lower()
            score = sum(1 for term in normalized.split() if term in haystack)
            if score <= 0:
                continue
            hits.append(
                KnowledgeHit(
                    document=doc,
                    score=float(score),
                    snippet=doc.text.splitlines()[-1],
                )
            )
        return sorted(hits, key=lambda hit: (-hit.score, hit.document.path))[:limit]

    def read(self, path: str) -> KnowledgeDocument:
        try:
            return self.documents[path]
        except KeyError as exc:
            raise FileNotFoundError(path) from exc


class FakeLongTermMemory:
    def read_profile(self) -> str:
        return "- Người dùng thích được giải thích chi tiết bằng tiếng Việt."


class TraceLLM:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        user_msg: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        inject_persona: bool = True,
    ) -> LLMResult:
        self.calls.append(
            {
                "user_msg": user_msg,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "inject_persona": inject_persona,
            }
        )
        return LLMResult(
            text=self.response_text,
            prompt=user_msg,
            n_prompt_tokens=max(1, len(user_msg.split())),
            n_completion_tokens=max(1, len(self.response_text.split())),
            ttft_ms=1.0,
            total_latency_ms=2.0,
            tokens_per_second=100.0,
        )


class DisabledKnowledgeSearchTool:
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="knowledge.search",
            description="Disabled test knowledge search.",
            input_schema=object_schema(
                properties={
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                required=["query"],
            ),
            side_effect=SideEffectLevel.READ_ONLY,
            enabled=False,
        )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(name=self.spec.name, ok=True, content="")


def load_cases(path: Path, limit: int | None = None) -> list[RuntimeEvalCase]:
    cases: list[RuntimeEvalCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            try:
                route = RuntimeRoute(str(payload["expected_route"]))
                case = RuntimeEvalCase(
                    case_id=str(payload["id"]),
                    category=str(payload["category"]),
                    text=str(payload["input"]),
                    expected_route=route,
                    should_block=bool(payload["should_block"]),
                    scenario=str(payload.get("scenario", "default")),
                    metadata=dict(payload.get("metadata", {})),
                    expected_tool=payload.get("expected_tool"),
                    requires_tool=payload.get("requires_tool"),
                    requires_llm=payload.get("requires_llm"),
                    requires_citation=bool(payload.get("requires_citation", False)),
                    expected_guardrail_reason=payload.get("expected_guardrail_reason"),
                    expected_citation_paths=tuple(payload.get("expected_citation_paths", ())),
                    expected_knowledge_hit_paths=tuple(
                        payload.get("expected_knowledge_hit_paths", ())
                    ),
                    expect_no_knowledge_hits=bool(payload.get("expect_no_knowledge_hits", False)),
                    expected_response_contains=tuple(
                        payload.get("expected_response_contains", ())
                    ),
                    fake_llm_response=str(
                        payload.get(
                            "fake_llm_response",
                            "Mình là SoCa, trợ lý tiếng Việt.",
                        )
                    ),
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{line_no} missing field: {exc}") from exc
            cases.append(case)
            if limit is not None and len(cases) >= limit:
                break
    if not cases:
        raise ValueError(f"No eval cases loaded from {path}")
    return cases


def build_runtime(case: RuntimeEvalCase) -> AssistantRuntime:
    knowledge_source = EvalKnowledgeSource(empty=case.scenario == "knowledge_empty")
    tools = [
        LocalTimeTool(now_fn=lambda: FIXED_NOW),
        KnowledgeReadTool(knowledge_source),
    ]
    if case.scenario == "knowledge_disabled":
        tools.append(DisabledKnowledgeSearchTool())
    else:
        tools.append(KnowledgeSearchTool(knowledge_source))

    session = SessionMemory()
    memory_builder = MemoryContextBuilder(
        long_term=FakeLongTermMemory(),
        session=session,
    )
    return AssistantRuntime(
        llm=TraceLLM(case.fake_llm_response),
        tool_runtime=ToolRuntime(tools),
        knowledge_builder=KnowledgeContextBuilder(knowledge_source),
        memory_builder=memory_builder,
        options=RuntimeOptions(max_tokens=96, temperature=0.0),
    )


def _trace_value(result, attr: str, default):
    trace = result.trace
    if trace is None:
        return default
    return getattr(trace, attr, default)


def evaluate_case(case: RuntimeEvalCase) -> RuntimeEvalSample:
    runtime = build_runtime(case)
    result = runtime.run_text_turn(case.text, metadata=case.metadata)
    trace = result.trace

    tool_calls = tuple(call.name for call in _trace_value(result, "tool_calls", ()))
    tool_results = tuple(
        {
            "name": tool_result.name,
            "ok": tool_result.ok,
            "error": tool_result.error,
        }
        for tool_result in _trace_value(result, "tool_results", ())
    )
    guardrail_reasons = tuple(
        event.reason
        for event in _trace_value(result, "guardrail_events", ())
        if event.reason
    )
    citation_paths = tuple(citation.path for citation in result.citations)
    knowledge_hit_paths = tuple(
        hit.document.path for hit in _trace_value(result, "knowledge_hits", ())
    )
    stage_latencies_ms = dict(_trace_value(result, "stage_latencies_ms", {}))
    used_tool = bool(trace and trace.used_tool)
    used_llm = bool(trace and trace.used_llm)

    checks: dict[str, bool] = {
        "route": result.route == case.expected_route,
        "blocked": result.blocked is case.should_block,
    }

    if case.expected_tool is not None:
        checks["expected_tool"] = case.expected_tool in tool_calls
    if case.requires_tool is not None:
        checks["tool_usage"] = used_tool is case.requires_tool
    if case.requires_llm is not None:
        checks["llm_usage"] = used_llm is case.requires_llm
    if case.requires_citation:
        checks["citation_present"] = bool(citation_paths)
    if case.expected_guardrail_reason is not None:
        checks["guardrail_reason"] = case.expected_guardrail_reason in guardrail_reasons
    if case.expected_citation_paths:
        checks["citation_paths"] = all(path in citation_paths for path in case.expected_citation_paths)
    if case.expected_knowledge_hit_paths:
        checks["knowledge_hit_paths"] = all(
            path in knowledge_hit_paths for path in case.expected_knowledge_hit_paths
        )
    if case.expect_no_knowledge_hits:
        checks["knowledge_hit_paths"] = not knowledge_hit_paths
    if case.expected_response_contains:
        checks["response_contains"] = all(
            text in result.response_text for text in case.expected_response_contains
        )

    return RuntimeEvalSample(
        case=case,
        response_text=result.response_text,
        route=result.route.value,
        blocked=result.blocked,
        tool_calls=tool_calls,
        tool_results=tool_results,
        guardrail_reasons=guardrail_reasons,
        citation_paths=citation_paths,
        knowledge_hit_paths=knowledge_hit_paths,
        used_tool=used_tool,
        used_llm=used_llm,
        stage_latencies_ms=stage_latencies_ms,
        checks=checks,
    )


def _rate(flags: Sequence[bool]) -> float | None:
    if not flags:
        return None
    return sum(flags) / len(flags)


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(int(len(sorted_values) * 0.95), len(sorted_values) - 1)
    return sorted_values[index]


def summarize(samples: Sequence[RuntimeEvalSample]) -> dict[str, Any]:
    totals = [sum(sample.stage_latencies_ms.values()) for sample in samples]
    sensitive_categories = {
        "guardrail_private_path",
        "guardrail_prompt_extraction",
        "guardrail_realtime_unsupported",
    }
    sensitive_samples = [
        sample for sample in samples if sample.case.category in sensitive_categories
    ]
    disabled_samples = [
        sample for sample in samples if sample.case.category == "disabled_tool"
    ]
    citation_samples = [
        sample
        for sample in samples
        if sample.case.requires_citation or sample.case.expected_citation_paths
    ]

    return {
        "n_cases": len(samples),
        "pass_rate": _rate([sample.passed for sample in samples]),
        "route_accuracy": _rate([sample.checks["route"] for sample in samples]),
        "blocked_accuracy": _rate([sample.checks["blocked"] for sample in samples]),
        "tool_selection_accuracy": _rate(
            [
                sample.checks["expected_tool"]
                for sample in samples
                if "expected_tool" in sample.checks
            ]
        ),
        "llm_usage_accuracy": _rate(
            [
                sample.checks["llm_usage"]
                for sample in samples
                if "llm_usage" in sample.checks
            ]
        ),
        "citation_pass_rate": _rate(
            [
                sample.checks.get("citation_present", True)
                and sample.checks.get("citation_paths", True)
                for sample in citation_samples
            ]
        ),
        "unsafe_to_llm_rate": _rate([sample.used_llm for sample in sensitive_samples]),
        "disabled_tool_execution_rate": _rate(
            [bool(sample.tool_results) for sample in disabled_samples]
        ),
        "latency_ms": {
            "mean": mean(totals) if totals else 0.0,
            "median": median(totals) if totals else 0.0,
            "p95": _p95(totals),
            "max": max(totals) if totals else 0.0,
        },
    }


def sample_to_dict(sample: RuntimeEvalSample) -> dict[str, Any]:
    return {
        "id": sample.case.case_id,
        "category": sample.case.category,
        "input": sample.case.text,
        "scenario": sample.case.scenario,
        "expected_route": sample.case.expected_route.value,
        "route": sample.route,
        "blocked": sample.blocked,
        "used_tool": sample.used_tool,
        "used_llm": sample.used_llm,
        "tool_calls": list(sample.tool_calls),
        "tool_results": list(sample.tool_results),
        "guardrail_reasons": list(sample.guardrail_reasons),
        "citation_paths": list(sample.citation_paths),
        "knowledge_hit_paths": list(sample.knowledge_hit_paths),
        "response_text": sample.response_text,
        "stage_latencies_ms": sample.stage_latencies_ms,
        "checks": sample.checks,
        "passed": sample.passed,
    }


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# SoCa AssistantRuntime Eval",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        if isinstance(value, dict):
            continue
        if value is None:
            text = "n/a"
        elif isinstance(value, float):
            text = f"{value:.1%}"
        else:
            text = str(value)
        lines.append(f"| {key} | {text} |")

    latency = summary["latency_ms"]
    lines.extend(
        [
            f"| latency_mean_ms | {latency['mean']:.2f} |",
            f"| latency_p95_ms | {latency['p95']:.2f} |",
            "",
            "## Cases",
            "",
            "| ID | Category | Route | Pass | Tool Calls | Guardrail Reasons | Citations |",
            "|---|---|---|---:|---|---|---|",
        ]
    )
    for sample in report["samples"]:
        lines.append(
            f"| {sample['id']} | {sample['category']} | {sample['route']} | "
            f"{'yes' if sample['passed'] else 'no'} | "
            f"{', '.join(sample['tool_calls']) or '-'} | "
            f"{', '.join(sample['guardrail_reasons']) or '-'} | "
            f"{', '.join(sample['citation_paths']) or '-'} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary_table(report: dict[str, Any]) -> None:
    summary = report["summary"]
    table = Table(title="SoCa AssistantRuntime Eval")
    table.add_column("Cases", justify="right")
    table.add_column("Pass", justify="right")
    table.add_column("Route", justify="right")
    table.add_column("Tool", justify="right")
    table.add_column("LLM usage", justify="right")
    table.add_column("Citation", justify="right")
    table.add_column("Unsafe->LLM", justify="right")
    table.add_column("Disabled exec", justify="right")
    table.add_column("p95 ms", justify="right")

    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1%}"

    table.add_row(
        str(summary["n_cases"]),
        fmt(summary["pass_rate"]),
        fmt(summary["route_accuracy"]),
        fmt(summary["tool_selection_accuracy"]),
        fmt(summary["llm_usage_accuracy"]),
        fmt(summary["citation_pass_rate"]),
        fmt(summary["unsafe_to_llm_rate"]),
        fmt(summary["disabled_tool_execution_rate"]),
        f"{summary['latency_ms']['p95']:.2f}",
    )
    console.print(table)


def run_eval(cases: Sequence[RuntimeEvalCase]) -> dict[str, Any]:
    samples = [
        evaluate_case(case)
        for case in track(cases, description="Evaluating AssistantRuntime...")
    ]
    return {
        "suite": "assistant_runtime",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": summarize(samples),
        "samples": [sample_to_dict(sample) for sample in samples],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate SoCa AssistantRuntime behavior.")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fail-on-regression", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = load_cases(args.prompts, limit=args.limit)
    report = run_eval(cases)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = args.output_dir / f"assistant_runtime_eval_{timestamp}.json"
    md_path = args.output_dir / f"assistant_runtime_eval_{timestamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(md_path, report)

    print_summary_table(report)
    console.print(f"\n[green]Saved[/green] {json_path}")
    console.print(f"[green]Saved[/green] {md_path}")

    if args.fail_on_regression and report["summary"]["pass_rate"] != 1.0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
