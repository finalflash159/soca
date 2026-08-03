"""Capture the current core/working/archive memory access policy.

This is a model-free policy harness. It uses generated, non-personal memory
hits only to measure whether the caller requests archive access and what the
prompt/token cost is; it is not a memory quality benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from soca.knowledge import KnowledgeDocument, KnowledgeHit
from soca.memory import MemoryContextBuilder
from soca.memory.base import MemoryRetrievalResult
from soca.memory.working import approximate_tokens


class _SyntheticMemory:
    def __init__(self, relevant_queries: set[str]) -> None:
        self.relevant_queries = relevant_queries
        self.core_reads = 0
        self.archive_reads = 0

    def read_core(self) -> str:
        self.core_reads += 1
        return "Approved core preference: trả lời bằng tiếng Việt, có cấu trúc rõ ràng."

    def retrieve_archive(self, query: str) -> MemoryRetrievalResult:
        self.archive_reads += 1
        if query not in self.relevant_queries:
            return MemoryRetrievalResult(text="", mode="retrieved", evidence_reason="no_hits")
        document = KnowledgeDocument(
            id="synthetic-policy-hit",
            path="memory/policy-harness.md",
            title="Synthetic policy hit",
            text=query,
        )
        hit = KnowledgeHit(
            document=document,
            score=0.95,
            snippet=query,
            line_start=1,
            line_end=1,
            retrieval_backend="synthetic-policy-harness",
        )
        return MemoryRetrievalResult(
            text="",
            hits=(hit,),
            mode="retrieved",
            evidence_status="supported",
            evidence_reason="synthetic_relevant_case",
            top_relevance=0.95,
        )


class _SyntheticSession:
    def render(self) -> str:
        return "Recent conversation:\nUser: đang kiểm tra memory policy\nAssistant: tiếp tục kiểm tra."


def load_cases(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    families: dict[str, str] = {}
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            row_id, family, split = row.get("id"), row.get("family"), row.get("split")
            if (
                not isinstance(row_id, str)
                or not row_id.strip()
                or row_id in ids
                or not isinstance(family, str)
                or split not in {"train", "validation", "test"}
                or not isinstance(row.get("query"), str)
                or not row["query"].strip()
                or not isinstance(row.get("expected_policy"), str)
                or not isinstance(row.get("archive_query"), bool)
            ):
                raise ValueError(f"{path}:{line_number}: invalid memory policy row")
            if family in families and families[family] != split:
                raise ValueError(f"{path}:{line_number}: family crosses splits: {family}")
            ids.add(row_id)
            families[family] = str(split)
            rows.append(row)
    if not rows or {row["split"] for row in rows} != {"train", "validation", "test"}:
        raise ValueError(f"{path}: memory policy dataset needs all three splits")
    return tuple(rows)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * percentile) - 1)]


def run(dataset: Path, output: Path) -> dict[str, Any]:
    rows = load_cases(dataset)
    relevant_queries = {
        row["query"]
        for row in rows
        if row["archive_query"] and not str(row["expected_policy"]).endswith("_empty")
    }
    source = _SyntheticMemory(relevant_queries)
    builder = MemoryContextBuilder(source, session=_SyntheticSession(), max_chars=64_000)
    records: list[dict[str, Any]] = []
    for row in rows:
        started = time.perf_counter()
        context = builder.build(row["query"], include_archive=row["archive_query"])
        records.append(
            {
                "id": row["id"],
                "expected_policy": row["expected_policy"],
                "archive_requested": row["archive_query"],
                "archive_hit_count": len(context.hits),
                "prompt_tokens": approximate_tokens(context.prompt_text),
                "latency_ms": (time.perf_counter() - started) * 1000,
                "evidence_status": context.evidence_status,
            }
        )
    prompt_tokens = [float(record["prompt_tokens"]) for record in records]
    latencies = [float(record["latency_ms"]) for record in records]
    archive_cases = [row for row in rows if row["archive_query"]]
    expected_archive = sum(row["archive_query"] for row in rows)
    actual_archive = source.archive_reads
    injected = [record for record in records if record["archive_hit_count"] > 0]
    precise = sum(record["id"] in {row["id"] for row in archive_cases if not str(row["expected_policy"]).endswith("_empty")} for record in injected)
    result = {
        "dataset": str(dataset),
        "case_count": len(rows),
        "core_reads": source.core_reads,
        "archive_search_rate": {
            "actual_calls": actual_archive,
            "expected_on_demand_calls": expected_archive,
            "over_all_rows": actual_archive / len(rows),
            "over_archive_cases": actual_archive / len(archive_cases) if archive_cases else 0.0,
        },
        "injected_memory_precision": {
            "accepted_injected_cases": len(injected),
            "relevant_injected_cases": precise,
            "precision": precise / len(injected) if injected else 1.0,
        },
        "prompt_tokens": {
            "mean": statistics.fmean(prompt_tokens) if prompt_tokens else 0.0,
            "p50": _percentile(prompt_tokens, 0.50),
            "p95": _percentile(prompt_tokens, 0.95),
        },
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p95": _percentile(latencies, 0.95),
        },
        "answer_delta": {
            "status": "not_measured",
            "reason": "model-free policy harness; requires paired answer-LLM run",
        },
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure core/working/archive memory access policy.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
