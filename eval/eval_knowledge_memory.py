"""Evaluate SoCa knowledge and memory context assembly.

This suite is model-free. It checks retrieval hits, citation formatting,
path-safety policy, untrusted note warnings, and memory budget behavior before
those contexts ever reach the LLM.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any

from rich.console import Console
from rich.progress import track
from rich.table import Table

from soca.core.guardrails import (
    GuardrailAction,
    GuardrailStage,
    check_knowledge_read_path,
    check_untrusted_text,
)
from soca.knowledge import KnowledgeContextBuilder, KnowledgeDocument, KnowledgeHit
from soca.memory import MemoryContextBuilder, SessionMemory

console = Console()
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_PATH = REPO_ROOT / "eval" / "prompts" / "knowledge_memory_vi.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "eval" / "results"


@dataclass(frozen=True)
class KnowledgeMemoryCase:
    case_id: str
    category: str
    operation: str
    query: str = ""
    path: str = ""
    text: str = ""
    expected_hit_paths: tuple[str, ...] = ()
    requires_citation: bool = False
    expected_prompt_contains: tuple[str, ...] = ()
    forbidden_prompt_contains: tuple[str, ...] = ()
    expected_block_reason: str | None = None
    expected_warning_reason: str | None = None
    expected_allow: bool = False
    max_prompt_chars: int | None = None
    session_turn_count: int = 0
    expected_max_turns: int | None = None
    max_render_chars: int | None = None


@dataclass(frozen=True)
class KnowledgeMemorySample:
    case: KnowledgeMemoryCase
    checks: dict[str, bool]
    details: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


class EvalKnowledgeSource:
    def __init__(self) -> None:
        self.documents = {
            "wiki/dinh-duong/chat-dam.md": KnowledgeDocument(
                id="chat-dam",
                path="wiki/dinh-duong/chat-dam.md",
                title="Chất đạm",
                text=(
                    "# Chất đạm\n"
                    "Protein hỗ trợ duy trì cơ bắp và giúp no lâu. "
                    "Nguồn gợi ý: trứng, cá, đậu phụ và sữa chua."
                ),
                tags=("dinh-duong", "protein"),
            ),
            "wiki/dinh-duong/bua-sang.md": KnowledgeDocument(
                id="bua-sang",
                path="wiki/dinh-duong/bua-sang.md",
                title="Bữa sáng cân bằng",
                text=(
                    "# Bữa sáng cân bằng\n"
                    "Một bữa sáng cân bằng nên có chất đạm, tinh bột chậm, "
                    "rau hoặc trái cây để giữ năng lượng ổn định."
                ),
                tags=("dinh-duong", "bua-sang"),
            ),
        }

    def search(self, query: str, limit: int = 5) -> list[KnowledgeHit]:
        normalized_terms = [term for term in query.lower().split() if term]
        hits: list[KnowledgeHit] = []
        for doc in self.documents.values():
            haystack = f"{doc.title}\n{doc.text}\n{' '.join(doc.tags)}".lower()
            score = sum(1 for term in normalized_terms if term in haystack)
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
        return self.documents[path]


class FakeLongTermMemory:
    def read_profile(self) -> str:
        return (
            "- Người dùng thích giải thích chi tiết bằng tiếng Việt.\n"
            "- Người dùng thích các bước rõ ràng và có ví dụ ngắn."
        )


def load_cases(path: Path, limit: int | None = None) -> list[KnowledgeMemoryCase]:
    cases: list[KnowledgeMemoryCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            try:
                cases.append(
                    KnowledgeMemoryCase(
                        case_id=str(payload["id"]),
                        category=str(payload["category"]),
                        operation=str(payload["operation"]),
                        query=str(payload.get("query", "")),
                        path=str(payload.get("path", "")),
                        text=str(payload.get("text", "")),
                        expected_hit_paths=tuple(payload.get("expected_hit_paths", ())),
                        requires_citation=bool(payload.get("requires_citation", False)),
                        expected_prompt_contains=tuple(
                            payload.get("expected_prompt_contains", ())
                        ),
                        forbidden_prompt_contains=tuple(
                            payload.get("forbidden_prompt_contains", ())
                        ),
                        expected_block_reason=payload.get("expected_block_reason"),
                        expected_warning_reason=payload.get("expected_warning_reason"),
                        expected_allow=bool(payload.get("expected_allow", False)),
                        max_prompt_chars=payload.get("max_prompt_chars"),
                        session_turn_count=int(payload.get("session_turn_count", 0)),
                        expected_max_turns=payload.get("expected_max_turns"),
                        max_render_chars=payload.get("max_render_chars"),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{line_no} missing field: {exc}") from exc
            if limit is not None and len(cases) >= limit:
                break
    if not cases:
        raise ValueError(f"No eval cases loaded from {path}")
    return cases


def evaluate_knowledge_context(case: KnowledgeMemoryCase) -> KnowledgeMemorySample:
    started = time.perf_counter()
    context = KnowledgeContextBuilder(EvalKnowledgeSource()).build(case.query)
    hit_paths = tuple(hit.document.path for hit in context.hits)
    citation_paths = tuple(citation.path for citation in context.citations)
    checks = {
        "hit_paths": all(path in hit_paths for path in case.expected_hit_paths),
        "citation_required": (not case.requires_citation) or bool(citation_paths),
        "prompt_contains": all(text in context.prompt_text for text in case.expected_prompt_contains),
        "prompt_forbidden": all(
            text not in context.prompt_text for text in case.forbidden_prompt_contains
        ),
    }
    if case.max_prompt_chars is not None:
        checks["prompt_budget"] = len(context.prompt_text) <= case.max_prompt_chars
    if not case.expected_hit_paths:
        checks["no_unexpected_hits"] = not hit_paths

    return KnowledgeMemorySample(
        case=case,
        checks=checks,
        details={
            "hit_paths": list(hit_paths),
            "citation_paths": list(citation_paths),
            "prompt_chars": len(context.prompt_text),
        },
        latency_ms=(time.perf_counter() - started) * 1000,
    )


def evaluate_path_guard(case: KnowledgeMemoryCase) -> KnowledgeMemorySample:
    started = time.perf_counter()
    event = check_knowledge_read_path(case.path)
    checks = {
        "allow": event.action == GuardrailAction.ALLOW
        if case.expected_allow
        else event.action == GuardrailAction.BLOCK,
    }
    if case.expected_block_reason is not None:
        checks["block_reason"] = event.reason == case.expected_block_reason

    return KnowledgeMemorySample(
        case=case,
        checks=checks,
        details={
            "action": event.action.value,
            "reason": event.reason,
            "metadata": event.metadata,
        },
        latency_ms=(time.perf_counter() - started) * 1000,
    )


def evaluate_untrusted_text(case: KnowledgeMemoryCase) -> KnowledgeMemorySample:
    started = time.perf_counter()
    event = check_untrusted_text(case.text, stage=GuardrailStage.RETRIEVAL)
    checks = {
        "warning": event.action == GuardrailAction.WARN,
    }
    if case.expected_warning_reason is not None:
        checks["warning_reason"] = event.reason == case.expected_warning_reason
    return KnowledgeMemorySample(
        case=case,
        checks=checks,
        details={"action": event.action.value, "reason": event.reason},
        latency_ms=(time.perf_counter() - started) * 1000,
    )


def evaluate_memory_context(case: KnowledgeMemoryCase) -> KnowledgeMemorySample:
    started = time.perf_counter()
    session = SessionMemory()
    session.append("user", "Hôm nay tôi muốn ăn gì cho đủ năng lượng?")
    session.append("assistant", "Bạn có thể bắt đầu với bữa sáng có protein và trái cây.")
    context = MemoryContextBuilder(long_term=FakeLongTermMemory(), session=session).build()
    checks = {
        "prompt_contains": all(text in context.prompt_text for text in case.expected_prompt_contains),
        "prompt_forbidden": all(
            text not in context.prompt_text for text in case.forbidden_prompt_contains
        ),
    }
    if case.max_prompt_chars is not None:
        checks["prompt_budget"] = len(context.prompt_text) <= case.max_prompt_chars

    return KnowledgeMemorySample(
        case=case,
        checks=checks,
        details={
            "prompt_chars": len(context.prompt_text),
            "profile_chars": len(context.profile_text),
            "session_chars": len(context.session_text),
        },
        latency_ms=(time.perf_counter() - started) * 1000,
    )


def evaluate_session_window(case: KnowledgeMemoryCase) -> KnowledgeMemorySample:
    started = time.perf_counter()
    max_render_chars = case.max_render_chars or 420
    session = SessionMemory(max_turns=case.expected_max_turns or 6, max_chars=max_render_chars)
    for index in range(case.session_turn_count):
        role = "user" if index % 2 == 0 else "assistant"
        session.append(role, f"lượt hội thoại số {index}")
    rendered = session.render()
    checks = {
        "turn_count": len(session.turns) <= (case.expected_max_turns or len(session.turns)),
        "render_budget": len(rendered) <= max_render_chars,
    }
    return KnowledgeMemorySample(
        case=case,
        checks=checks,
        details={
            "turn_count": len(session.turns),
            "render_chars": len(rendered),
        },
        latency_ms=(time.perf_counter() - started) * 1000,
    )


def evaluate_case(case: KnowledgeMemoryCase) -> KnowledgeMemorySample:
    if case.operation == "knowledge_context":
        return evaluate_knowledge_context(case)
    if case.operation == "path_guard":
        return evaluate_path_guard(case)
    if case.operation == "untrusted_text":
        return evaluate_untrusted_text(case)
    if case.operation == "memory_context":
        return evaluate_memory_context(case)
    if case.operation == "session_window":
        return evaluate_session_window(case)
    raise ValueError(f"Unknown knowledge/memory operation: {case.operation}")


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


def summarize(samples: Sequence[KnowledgeMemorySample]) -> dict[str, Any]:
    knowledge_samples = [sample for sample in samples if sample.case.operation == "knowledge_context"]
    path_samples = [sample for sample in samples if sample.case.operation == "path_guard"]
    memory_samples = [
        sample
        for sample in samples
        if sample.case.operation in {"memory_context", "session_window"}
    ]
    citation_samples = [sample for sample in knowledge_samples if sample.case.requires_citation]
    budget_samples = [
        sample
        for sample in samples
        if "prompt_budget" in sample.checks or "render_budget" in sample.checks
    ]
    latencies = [sample.latency_ms for sample in samples]

    return {
        "n_cases": len(samples),
        "pass_rate": _rate([sample.passed for sample in samples]),
        "retrieval_hit_rate": _rate(
            [sample.checks.get("hit_paths", True) for sample in knowledge_samples]
        ),
        "citation_pass_rate": _rate(
            [sample.checks.get("citation_required", True) for sample in citation_samples]
        ),
        "path_guard_pass_rate": _rate([sample.passed for sample in path_samples]),
        "memory_context_pass_rate": _rate([sample.passed for sample in memory_samples]),
        "context_budget_violation_rate": _rate(
            [
                not sample.checks.get("prompt_budget", sample.checks.get("render_budget", True))
                for sample in budget_samples
            ]
        ),
        "latency_ms": {
            "mean": mean(latencies) if latencies else 0.0,
            "median": median(latencies) if latencies else 0.0,
            "p95": _p95(latencies),
            "max": max(latencies) if latencies else 0.0,
        },
    }


def sample_to_dict(sample: KnowledgeMemorySample) -> dict[str, Any]:
    return {
        "id": sample.case.case_id,
        "category": sample.case.category,
        "operation": sample.case.operation,
        "checks": sample.checks,
        "passed": sample.passed,
        "latency_ms": sample.latency_ms,
        "details": sample.details,
    }


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# SoCa Knowledge/Memory Eval",
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
            "| ID | Operation | Pass | Details |",
            "|---|---|---:|---|",
        ]
    )
    for sample in report["samples"]:
        detail_text = json.dumps(sample["details"], ensure_ascii=False, sort_keys=True)
        lines.append(
            f"| {sample['id']} | {sample['operation']} | "
            f"{'yes' if sample['passed'] else 'no'} | `{detail_text}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary_table(report: dict[str, Any]) -> None:
    summary = report["summary"]
    table = Table(title="SoCa Knowledge/Memory Eval")
    table.add_column("Cases", justify="right")
    table.add_column("Pass", justify="right")
    table.add_column("Retrieval", justify="right")
    table.add_column("Citation", justify="right")
    table.add_column("Path guard", justify="right")
    table.add_column("Memory", justify="right")
    table.add_column("Budget violation", justify="right")
    table.add_column("p95 ms", justify="right")

    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1%}"

    table.add_row(
        str(summary["n_cases"]),
        fmt(summary["pass_rate"]),
        fmt(summary["retrieval_hit_rate"]),
        fmt(summary["citation_pass_rate"]),
        fmt(summary["path_guard_pass_rate"]),
        fmt(summary["memory_context_pass_rate"]),
        fmt(summary["context_budget_violation_rate"]),
        f"{summary['latency_ms']['p95']:.2f}",
    )
    console.print(table)


def run_eval(cases: Sequence[KnowledgeMemoryCase]) -> dict[str, Any]:
    samples = [
        evaluate_case(case)
        for case in track(cases, description="Evaluating knowledge/memory...")
    ]
    return {
        "suite": "knowledge_memory",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": summarize(samples),
        "samples": [sample_to_dict(sample) for sample in samples],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate SoCa knowledge and memory contexts.")
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
    json_path = args.output_dir / f"knowledge_memory_eval_{timestamp}.json"
    md_path = args.output_dir / f"knowledge_memory_eval_{timestamp}.md"
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
