"""Measure answerable retrieval recall and unanswerable accepted evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from eval.eval_hybrid_retrieval import _build_source
from soca.knowledge.context import KnowledgeContextBuilder
from soca.knowledge.relevance import RelevancePolicy


def load_dataset(path: Path) -> tuple[dict[str, Any], ...]:
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
            relevant = row.get("relevant_paths")
            answerable = row.get("answerable")
            if (
                not isinstance(row_id, str)
                or not row_id.strip()
                or row_id in ids
                or not isinstance(family, str)
                or split not in {"train", "validation", "test"}
                or not isinstance(row.get("query"), str)
                or not row["query"].strip()
                or not isinstance(answerable, bool)
                or not isinstance(relevant, list)
                or any(not isinstance(path, str) or not path.startswith("wiki/") for path in relevant)
                or (answerable and not relevant)
                or (not answerable and relevant)
            ):
                raise ValueError(f"{path}:{line_number}: invalid grounding row")
            if family in families and families[family] != split:
                raise ValueError(f"{path}:{line_number}: family crosses splits: {family}")
            ids.add(row_id)
            families[family] = str(split)
            rows.append(row)
    if not rows or {row["split"] for row in rows} != {"train", "validation", "test"}:
        raise ValueError(f"{path}: grounding dataset needs all three splits")
    return tuple(rows)


def _wilson(successes: int, total: int) -> dict[str, float | int]:
    if total == 0:
        return {"successes": successes, "total": total, "rate": 0.0, "lower": 0.0, "upper": 0.0}
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    centre = rate + z * z / (2 * total)
    spread = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
    return {
        "successes": successes,
        "total": total,
        "rate": rate,
        "lower": max(0.0, (centre - spread) / denominator),
        "upper": min(1.0, (centre + spread) / denominator),
    }


def run_benchmark(
    *, vault: Path, dataset: Path, variant: str, backend: str, index_home: Path | None = None
) -> dict[str, Any]:
    rows = load_dataset(dataset)
    with tempfile.TemporaryDirectory(prefix="soca-grounding-p0-") as temporary_index:
        source = _build_source(
            variant=variant,
            backend=backend,
            vault=vault,
            index_home=index_home or Path(temporary_index),
            rrf_k=60,
        )
        builder = KnowledgeContextBuilder(
            source,
            max_hits=5,
            relevance_policy=RelevancePolicy.for_retrieval_mode(variant),
        )
        records: list[dict[str, Any]] = []
        for row in rows:
            started = time.perf_counter()
            retrieve = getattr(source, "retrieve", None)
            if callable(retrieve):
                batch = retrieve(row["query"], limit=5)
                raw_hits = tuple(batch.hits)
                context = builder.build_from_hits(
                    row["query"],
                    raw_hits,
                    diagnostics=batch.diagnostics,
                )
            else:
                raw_hits = tuple(source.search(row["query"], limit=5))
                context = builder.build_from_hits(row["query"], raw_hits)
            raw_paths = list(dict.fromkeys(hit.document.path for hit in raw_hits))
            accepted_paths = list(dict.fromkeys(hit.document.path for hit in context.hits))
            records.append(
                {
                    "id": row["id"],
                    "answerable": row["answerable"],
                    "raw_paths": raw_paths,
                    "accepted_paths": accepted_paths,
                    "evidence_status": context.evidence_status,
                    "evidence_reason": context.evidence_reason,
                    "retrieval_state": context.retrieval_state,
                    "query_coverage": context.query_coverage,
                    "score_separation": context.score_separation,
                    "sparse_top_score": context.sparse_top_score,
                    "dense_top_score": context.dense_top_score,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }
            )
    answerable = [
        (row, record)
        for row, record in zip(rows, records, strict=True)
        if row["answerable"]
    ]
    unanswerable = [
        record for row, record in zip(rows, records, strict=True) if not row["answerable"]
    ]
    raw_recall = sum(
        bool(set(record["raw_paths"][:5]) & set(row["relevant_paths"]))
        for row, record in answerable
    )
    recall = sum(
        bool(set(record["accepted_paths"][:5]) & set(row["relevant_paths"]))
        for row, record in answerable
    )
    false_evidence = sum(bool(record["accepted_paths"]) for record in unanswerable)
    latencies = sorted(float(record["latency_ms"]) for record in records)
    p95 = latencies[min(len(latencies) - 1, math.ceil(len(latencies) * 0.95) - 1)] if latencies else 0.0
    warm_latencies = sorted(float(record["latency_ms"]) for record in records[1:])
    warm_p95 = (
        warm_latencies[min(len(warm_latencies) - 1, math.ceil(len(warm_latencies) * 0.95) - 1)]
        if warm_latencies
        else 0.0
    )
    return {
        "dataset": str(dataset),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "vault": str(vault),
        "variant": variant,
        "backend": backend,
        "case_count": len(rows),
        "answerable_retrieval_recall_at_5": _wilson(recall, len(answerable)),
        "answerable_accepted_evidence_recall_at_5": _wilson(recall, len(answerable)),
        "answerable_raw_retrieval_recall_at_5": _wilson(raw_recall, len(answerable)),
        "unanswerable_false_evidence_rate": _wilson(false_evidence, len(unanswerable)),
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p95": p95,
            "first_query": float(records[0]["latency_ms"]) if records else 0.0,
            "warm_p95": warm_p95,
        },
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval grounding/abstention over a labelled vault.")
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--variant", choices=("cached_sparse", "chunk_sparse", "dense", "hybrid"), default="cached_sparse")
    parser.add_argument("--backend", choices=("fastembed", "model2vec"), default="fastembed")
    parser.add_argument("--index-home", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_benchmark(
        vault=args.vault.expanduser().resolve(),
        dataset=args.dataset,
        variant=args.variant,
        backend=args.backend,
        index_home=args.index_home,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
