from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path

from eval.eval_hybrid_retrieval import build_embedding_model
from eval.retrieval_bakeoff import DEFAULT_DATA_ROOT, _load_dataset
from eval.retrieval_benchmark_data import production_chunks


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction + 0.999999) - 1))
    return ordered[index]


def _output_path(argument: Path | None) -> Path:
    if argument is not None:
        return argument
    run_dir = os.environ.get("SOCA_BENCHMARK_RUN_DIR")
    if not run_dir:
        raise ValueError("--output or SOCA_BENCHMARK_RUN_DIR is required")
    return Path(run_dir) / "resource.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated embedding resource probe.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--dataset", default="tvpl")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--document-limit", type=int, default=128)
    parser.add_argument("--query-repeats", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.document_limit < 1 or args.query_repeats < 1:
        raise ValueError("probe limits must be positive")

    dataset = _load_dataset(args.dataset, args.data_root)
    chunks, _ = production_chunks(dataset)
    query = dataset.queries[sorted(dataset.queries)[0]]
    texts = tuple(chunk.text for chunk in chunks[: args.document_limit])
    baseline_rss = _rss_bytes()
    started = time.perf_counter()
    model = build_embedding_model(args.candidate)
    load_seconds = time.perf_counter() - started
    loaded_rss = _rss_bytes()
    started = time.perf_counter()
    vectors = model.embed_documents(texts)
    document_seconds = time.perf_counter() - started
    document_rss = _rss_bytes()
    query_latencies: list[float] = []
    for _ in range(args.query_repeats):
        started_ns = time.perf_counter_ns()
        model.embed_query(query)
        query_latencies.append((time.perf_counter_ns() - started_ns) / 1_000_000)
    payload = {
        "schema_version": 1,
        "candidate": args.candidate,
        "dataset": args.dataset,
        "document_count": len(texts),
        "dimension": int(vectors.shape[1]),
        "load_seconds": load_seconds,
        "document_seconds": document_seconds,
        "document_throughput_per_second": len(texts) / document_seconds,
        "query_repeats": args.query_repeats,
        "query_mean_ms": statistics.fmean(query_latencies),
        "query_p50_ms": _percentile(query_latencies, 0.50),
        "query_p95_ms": _percentile(query_latencies, 0.95),
        "baseline_rss_bytes": baseline_rss,
        "loaded_rss_bytes": loaded_rss,
        "document_peak_rss_bytes": document_rss,
        "rss_delta_after_load_bytes": max(0, loaded_rss - baseline_rss),
        "rss_delta_after_documents_bytes": max(0, document_rss - baseline_rss),
    }
    output = _output_path(args.output)
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(output)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
