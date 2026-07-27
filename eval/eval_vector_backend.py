from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

SearchFunction = Callable[[np.ndarray], np.ndarray]

BACKENDS = (
    "numpy-current",
    "numpy-exact",
    "faiss-flat",
    "faiss-hnsw",
    "usearch-hnsw",
)


def normalize(matrix: np.ndarray) -> np.ndarray:
    normalized = np.ascontiguousarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.all(np.isfinite(norms)):
        raise ValueError("cannot normalize zero or non-finite vectors")
    normalized /= norms
    return normalized


def stable_exact_top_k(scores: np.ndarray, *, limit: int) -> np.ndarray:
    """Return exact top-k with deterministic row-id ordering at the kth boundary."""
    row_count = len(scores)
    if limit < 1:
        raise ValueError("limit must be positive")
    if row_count == 0:
        return np.empty(0, dtype=np.int64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")
    selected_limit = min(limit, row_count)
    row_ids = np.arange(row_count, dtype=np.int64)
    if selected_limit == row_count:
        return row_ids[np.lexsort((row_ids, -scores))]

    threshold_index = row_count - selected_limit
    threshold = np.partition(scores, threshold_index)[threshold_index]
    higher = np.flatnonzero(scores > threshold)
    boundary = np.flatnonzero(scores == threshold)
    remaining = selected_limit - len(higher)
    selected = np.concatenate((higher, boundary[:remaining]))
    return selected[np.lexsort((selected, -scores[selected]))]


def current_numpy_top_k(scores: np.ndarray, *, limit: int) -> np.ndarray:
    selected_limit = min(limit, len(scores))
    order = sorted(
        range(len(scores)),
        key=lambda index: (-float(scores[index]), index),
    )[:selected_limit]
    return np.asarray(order, dtype=np.int64)


def latency_summary(samples: Sequence[float]) -> dict[str, float]:
    if not samples:
        raise ValueError("latency samples must be non-empty")
    ordered = sorted(samples)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)
        return ordered[index]

    return {
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
    }


def recall_at_k(expected: np.ndarray, actual: np.ndarray, *, k: int) -> float:
    if len(expected) != len(actual):
        raise ValueError("expected and actual query counts differ")
    if not len(expected):
        raise ValueError("recall requires at least one query")
    return statistics.fmean(
        len(set(expected_row.tolist()) & set(actual_row.tolist())) / k
        for expected_row, actual_row in zip(expected, actual, strict=True)
    )


def measure_single_query(
    search: SearchFunction,
    queries: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    search(queries[0])
    durations: list[float] = []
    matches: list[np.ndarray] = []
    for query in queries:
        started = time.perf_counter_ns()
        match = np.asarray(search(query), dtype=np.int64)
        durations.append((time.perf_counter_ns() - started) / 1_000_000)
        matches.append(match)
    return latency_summary(durations), np.stack(matches)


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_metadata() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "working_tree_dirty": bool(status.strip())}


def serialized_size(save: Callable[[Path], None], *, suffix: str) -> int:
    with tempfile.TemporaryDirectory(prefix="soca-vector-backend-") as directory:
        path = Path(directory) / f"index{suffix}"
        save(path)
        return path.stat().st_size


def memory_preflight(
    *,
    size: int,
    dimension: int,
    query_count: int,
    backends: Sequence[str],
    max_memory_mib: int,
) -> dict[str, int]:
    vector_bytes = size * dimension * np.dtype(np.float32).itemsize
    query_bytes = query_count * dimension * np.dtype(np.float32).itemsize
    multiplier = 3 if any(backend.endswith("hnsw") for backend in backends) else 2
    estimated_peak_bytes = multiplier * vector_bytes + query_bytes
    limit_bytes = max_memory_mib * 1024 * 1024
    if estimated_peak_bytes > limit_bytes:
        raise MemoryError(
            "estimated benchmark peak exceeds --max-memory-mib: "
            f"{estimated_peak_bytes / 1024 / 1024:.1f} MiB > {max_memory_mib} MiB"
        )
    return {
        "vector_bytes": vector_bytes,
        "query_bytes": query_bytes,
        "estimated_peak_bytes": estimated_peak_bytes,
        "limit_bytes": limit_bytes,
    }


def benchmark_size(
    *,
    size: int,
    dimension: int,
    query_count: int,
    k: int,
    seed: int,
    backends: Sequence[str],
    hnsw_m: int,
    ef_construction: int,
    ef_search_values: Sequence[int],
    max_memory_mib: int,
) -> dict[str, Any]:
    preflight = memory_preflight(
        size=size,
        dimension=dimension,
        query_count=query_count,
        backends=backends,
        max_memory_mib=max_memory_mib,
    )
    rng = np.random.default_rng(seed + size + dimension)
    vectors = normalize(rng.standard_normal((size, dimension), dtype=np.float32))
    queries = normalize(rng.standard_normal((query_count, dimension), dtype=np.float32))
    oracle = np.stack(
        [stable_exact_top_k(vectors @ query, limit=k) for query in queries]
    )
    result: dict[str, Any] = {
        "size": size,
        "dimension": dimension,
        "preflight": preflight,
        "backends": {},
    }

    if "numpy-current" in backends:
        latency, matches = measure_single_query(
            lambda query: current_numpy_top_k(vectors @ query, limit=k),
            queries,
        )
        result["backends"]["numpy-current"] = {
            "latency": latency,
            "recall_at_k": recall_at_k(oracle, matches, k=k),
            "vector_bytes": vectors.nbytes,
        }

    if "numpy-exact" in backends:
        latency, matches = measure_single_query(
            lambda query: stable_exact_top_k(vectors @ query, limit=k),
            queries,
        )
        result["backends"]["numpy-exact"] = {
            "latency": latency,
            "recall_at_k": recall_at_k(oracle, matches, k=k),
            "vector_bytes": vectors.nbytes,
        }

    faiss = None
    if any(backend.startswith("faiss-") for backend in backends):
        try:
            import faiss as faiss_module
        except ImportError as exc:
            raise RuntimeError(
                "FAISS backend requested; install/provide faiss-cpu"
            ) from exc
        faiss = faiss_module

    if "faiss-flat" in backends:
        assert faiss is not None
        flat = faiss.IndexFlatIP(dimension)
        started = time.perf_counter_ns()
        flat.add(vectors)
        build_ms = (time.perf_counter_ns() - started) / 1_000_000
        latency, matches = measure_single_query(
            lambda query: flat.search(query.reshape(1, -1), k)[1][0],
            queries,
        )
        result["backends"]["faiss-flat"] = {
            "build_ms": build_ms,
            "latency": latency,
            "recall_at_k": recall_at_k(oracle, matches, k=k),
            "serialized_bytes": serialized_size(
                lambda path: faiss.write_index(flat, str(path)),
                suffix=".faiss",
            ),
        }

    if "faiss-hnsw" in backends:
        assert faiss is not None
        hnsw = faiss.IndexHNSWFlat(
            dimension,
            hnsw_m,
            faiss.METRIC_INNER_PRODUCT,
        )
        hnsw.hnsw.efConstruction = ef_construction
        started = time.perf_counter_ns()
        hnsw.add(vectors)
        build_ms = (time.perf_counter_ns() - started) / 1_000_000
        searches: dict[str, Any] = {}
        for ef_search in ef_search_values:
            hnsw.hnsw.efSearch = ef_search
            latency, matches = measure_single_query(
                lambda query: hnsw.search(query.reshape(1, -1), k)[1][0],
                queries,
            )
            searches[str(ef_search)] = {
                "latency": latency,
                "recall_at_k": recall_at_k(oracle, matches, k=k),
            }
        result["backends"]["faiss-hnsw"] = {
            "m": hnsw_m,
            "ef_construction": ef_construction,
            "build_ms": build_ms,
            "searches": searches,
            "serialized_bytes": serialized_size(
                lambda path: faiss.write_index(hnsw, str(path)),
                suffix=".faiss",
            ),
        }

    if "usearch-hnsw" in backends:
        try:
            from usearch.index import Index
        except ImportError as exc:
            raise RuntimeError(
                "USearch backend requested; install/provide usearch"
            ) from exc
        usearch_index = Index(
            ndim=dimension,
            metric="cos",
            dtype="f32",
            connectivity=hnsw_m,
            expansion_add=ef_construction,
            expansion_search=ef_search_values[0],
        )
        started = time.perf_counter_ns()
        usearch_index.add(np.arange(size, dtype=np.uint64), vectors)
        build_ms = (time.perf_counter_ns() - started) / 1_000_000
        searches = {}
        for ef_search in ef_search_values:
            usearch_index.expansion_search = ef_search
            latency, matches = measure_single_query(
                lambda query: usearch_index.search(query, k).keys,
                queries,
            )
            searches[str(ef_search)] = {
                "latency": latency,
                "recall_at_k": recall_at_k(oracle, matches, k=k),
            }
        result["backends"]["usearch-hnsw"] = {
            "m": hnsw_m,
            "ef_construction": ef_construction,
            "build_ms": build_ms,
            "searches": searches,
            "serialized_bytes": serialized_size(
                lambda path: usearch_index.save(str(path)),
                suffix=".usearch",
            ),
        }

    return result


def parse_positive_ints(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    if not values or any(value < 1 for value in values):
        raise ValueError(f"{name} values must be positive")
    return tuple(dict.fromkeys(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark exact and ANN vector-search backends for SoCa."
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=(1_000, 10_000, 50_000))
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument("--query-count", type=int, default=40)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_260_728)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=BACKENDS,
        default=("numpy-current", "numpy-exact"),
    )
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument(
        "--ef-search",
        type=int,
        nargs="+",
        default=(32, 64, 128, 256, 512, 1024),
    )
    parser.add_argument("--max-memory-mib", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sizes = parse_positive_ints(args.sizes, name="size")
    ef_search_values = parse_positive_ints(args.ef_search, name="ef-search")
    backends = tuple(dict.fromkeys(args.backends))
    if args.dimension < 1 or args.query_count < 1 or args.k < 1:
        raise ValueError("dimension, query-count, and k must be positive")
    if args.k > min(sizes):
        raise ValueError("k cannot exceed the smallest benchmark size")
    script_path = Path(__file__).resolve()
    report = {
        "status": "ok",
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            **git_metadata(),
            "script": str(script_path.relative_to(Path.cwd().resolve())),
            "script_sha256": file_sha256(script_path),
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                "numpy": package_version("numpy"),
                "faiss-cpu": package_version("faiss-cpu"),
                "usearch": package_version("usearch"),
            },
            "settings": {
                "sizes": sizes,
                "dimension": args.dimension,
                "query_count": args.query_count,
                "k": args.k,
                "seed": args.seed,
                "backends": backends,
                "hnsw_m": args.hnsw_m,
                "ef_construction": args.ef_construction,
                "ef_search": ef_search_values,
                "max_memory_mib": args.max_memory_mib,
                "singleton_queries": True,
                "dataset": "normalized_gaussian_float32",
            },
        },
        "results": [
            benchmark_size(
                size=size,
                dimension=args.dimension,
                query_count=args.query_count,
                k=args.k,
                seed=args.seed,
                backends=backends,
                hnsw_m=args.hnsw_m,
                ef_construction=args.ef_construction,
                ef_search_values=ef_search_values,
                max_memory_mib=args.max_memory_mib,
            )
            for size in sizes
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
