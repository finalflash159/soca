from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from statistics import mean
from typing import Any

from eval.embedding_candidates import build_eval_candidate
from soca.knowledge.base import KnowledgeSource
from soca.knowledge.cached_source import CachedMarkdownVaultKnowledgeSource
from soca.knowledge.hybrid_source import (
    DenseUnavailableError,
    HybridConfig,
    HybridKnowledgeSource,
)
from soca.knowledge.retrievers.dense import FastEmbedModel, Model2VecModel


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    slice_name: str
    query: str
    relevant_paths: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalSample:
    case_id: str
    slice_name: str
    retrieved_paths: tuple[str, ...]
    recall_at_5: float
    reciprocal_rank_at_10: float
    ndcg_at_10: float
    latency_ms: float


def _validate_path(path: str) -> None:
    if (
        not path.startswith("wiki/")
        or not path.endswith(".md")
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise ValueError(f"invalid relevant path: {path!r}")


def load_cases(path: Path) -> tuple[RetrievalCase, ...]:
    cases: list[RetrievalCase] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be an object")
            case_id = payload.get("id")
            slice_name = payload.get("slice")
            query = payload.get("query")
            relevant = payload.get("relevant_paths")
            if (
                not isinstance(case_id, str)
                or not case_id.strip()
                or not isinstance(slice_name, str)
                or not slice_name.strip()
                or not isinstance(query, str)
                or not query.strip()
                or not isinstance(relevant, list)
                or not relevant
                or any(not isinstance(item, str) for item in relevant)
            ):
                raise ValueError(f"{path}:{line_number} has invalid fields")
            if case_id in seen:
                raise ValueError(f"duplicate case id: {case_id}")
            seen.add(case_id)
            relevant_paths = tuple(relevant)
            if len(relevant_paths) != len(set(relevant_paths)):
                raise ValueError(f"{case_id}: relevant paths must be unique")
            for relevant_path in relevant_paths:
                _validate_path(relevant_path)
            cases.append(RetrievalCase(case_id, slice_name, query, relevant_paths))
    if not cases:
        raise ValueError("retrieval eval requires at least one case")
    return tuple(cases)


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], *, k: int) -> float:
    _validate_k(k)
    if not relevant:
        raise ValueError("relevant paths must be non-empty")
    unique_retrieved = tuple(dict.fromkeys(retrieved))
    return len(set(unique_retrieved[:k]) & set(relevant)) / len(set(relevant))


def reciprocal_rank_at_k(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    *,
    k: int,
) -> float:
    _validate_k(k)
    relevant_set = set(relevant)
    unique_retrieved = tuple(dict.fromkeys(retrieved))
    for rank, path in enumerate(unique_retrieved[:k], start=1):
        if path in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], *, k: int) -> float:
    _validate_k(k)
    relevant_set = set(relevant)
    unique_retrieved = tuple(dict.fromkeys(retrieved))
    dcg = sum(
        (1.0 / math.log2(rank + 1)) if path in relevant_set else 0.0
        for rank, path in enumerate(unique_retrieved[:k], start=1)
    )
    ideal_count = min(k, len(relevant_set))
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal if ideal else 0.0


def evaluate_source(
    source: KnowledgeSource,
    cases: Sequence[RetrievalCase],
) -> tuple[RetrievalSample, ...]:
    samples: list[RetrievalSample] = []
    for case in cases:
        started = time.perf_counter()
        hits = source.search(case.query, limit=50)
        latency_ms = (time.perf_counter() - started) * 1000
        paths = tuple(dict.fromkeys(hit.document.path for hit in hits))[:10]
        samples.append(
            RetrievalSample(
                case.case_id,
                case.slice_name,
                paths,
                recall_at_k(paths, case.relevant_paths, k=5),
                reciprocal_rank_at_k(paths, case.relevant_paths, k=10),
                ndcg_at_k(paths, case.relevant_paths, k=10),
                latency_ms,
            )
        )
    return tuple(samples)


def summarize(samples: Sequence[RetrievalSample]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot summarize empty retrieval samples")

    def metrics(group: Sequence[RetrievalSample]) -> dict[str, float]:
        ordered = sorted(sample.latency_ms for sample in group)
        p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
        return {
            "recall_at_5": mean(sample.recall_at_5 for sample in group),
            "mrr_at_10": mean(sample.reciprocal_rank_at_10 for sample in group),
            "ndcg_at_10": mean(sample.ndcg_at_10 for sample in group),
            "latency_mean_ms": mean(ordered),
            "latency_p95_ms": ordered[p95_index],
        }

    slices = sorted({sample.slice_name for sample in samples})
    return {
        "overall": metrics(samples),
        "by_slice": {
            slice_name: metrics([sample for sample in samples if sample.slice_name == slice_name])
            for slice_name in slices
        },
    }


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def build_embedding_model(backend: str):
    if backend == "fastembed":
        return FastEmbedModel()
    if backend == "model2vec":
        return Model2VecModel()
    return build_eval_candidate(backend)


def _measure_query_encoding(
    backend: str,
    cases: Sequence[RetrievalCase],
    *,
    repeats: int = 5,
) -> dict[str, float]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    try:
        model = build_embedding_model(backend)
        durations: list[float] = []
        for _ in range(repeats):
            for case in cases:
                started = time.perf_counter()
                model.embed_query(case.query)
                durations.append((time.perf_counter() - started) * 1000)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise DenseUnavailableError(f"dense backend {backend} cannot be benchmarked") from exc
    ordered = sorted(durations)
    p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return {"query_mean_ms": mean(ordered), "query_p95_ms": ordered[p95_index]}


def _build_source(
    *,
    variant: str,
    backend: str,
    vault: Path,
    index_home: Path,
    rrf_k: int,
) -> KnowledgeSource:
    common = {
        "root": vault,
        "index_home": index_home,
        "include_globs": ("wiki/**/*.md",),
    }
    if variant == "cached_sparse":
        return CachedMarkdownVaultKnowledgeSource(**common)
    model = None
    if variant != "chunk_sparse":
        try:
            model = build_embedding_model(backend)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            raise DenseUnavailableError(f"dense backend {backend} is unavailable") from exc
    return HybridKnowledgeSource(
        **common,
        model=model,
        config=HybridConfig(
            rrf_k=rrf_k,
            sparse_enabled=variant != "dense",
            dense_enabled=variant != "chunk_sparse",
            dense_failure_policy="raise" if variant in {"dense", "hybrid"} else "degrade",
        ),
    )


def _timed_evaluate(
    source_factory: Callable[[], KnowledgeSource],
    cases: Sequence[RetrievalCase],
) -> tuple[tuple[RetrievalSample, ...], float]:
    started = time.perf_counter()
    source = source_factory()
    samples = evaluate_source(source, cases)
    return samples, (time.perf_counter() - started) * 1000


def run_benchmark(
    *,
    vault: Path,
    cases: Sequence[RetrievalCase],
    variant: str,
    backend: str,
    rrf_k: int,
    warm_repeats: int,
) -> dict[str, Any]:
    if warm_repeats < 1:
        raise ValueError("warm_repeats must be positive")
    with tempfile.TemporaryDirectory(prefix="soca-rag-cold-") as cold_directory:
        cold_samples, cold_total_ms = _timed_evaluate(
            lambda: _build_source(
                variant=variant,
                backend=backend,
                vault=vault,
                index_home=Path(cold_directory),
                rrf_k=rrf_k,
            ),
            cases,
        )
    with tempfile.TemporaryDirectory(prefix="soca-rag-warm-") as warm_directory:
        warm_home = Path(warm_directory)
        _timed_evaluate(
            lambda: _build_source(
                variant=variant,
                backend=backend,
                vault=vault,
                index_home=warm_home,
                rrf_k=rrf_k,
            ),
            cases,
        )
        warm_samples: list[RetrievalSample] = []
        warm_totals: list[float] = []
        for _ in range(warm_repeats):
            samples, total_ms = _timed_evaluate(
                lambda: _build_source(
                    variant=variant,
                    backend=backend,
                    vault=vault,
                    index_home=warm_home,
                    rrf_k=rrf_k,
                ),
                cases,
            )
            warm_samples.extend(samples)
            warm_totals.append(total_ms)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "status": "ok",
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "commit": commit,
            "python": sys.version,
            "platform": platform.platform(),
            "case_count": len(cases),
            "variant": variant,
            "backend": backend,
            "rrf_k": rrf_k,
            "warm_repeats": warm_repeats,
            "packages": {
                name: _package_version(name)
                for name in (
                    "fastembed",
                    "model2vec",
                    "numpy",
                    "sentence-transformers",
                    "underthesea",
                )
            },
        },
        "cold": {"total_ms": cold_total_ms, "metrics": summarize(cold_samples)},
        "warm": {
            "total_mean_ms": mean(warm_totals),
            "metrics": summarize(warm_samples),
        },
        "encoding": (
            _measure_query_encoding(backend, cases) if variant in {"dense", "hybrid"} else None
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate SoCa hybrid retrieval.")
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=("cached_sparse", "chunk_sparse", "dense", "hybrid"),
        required=True,
    )
    parser.add_argument(
        "--backend",
        choices=("fastembed", "model2vec", "aiteamvn_bge_m3", "bkai_phobert_seg"),
        default="fastembed",
    )
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--warm-repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = run_benchmark(
            vault=args.vault.expanduser().resolve(),
            cases=load_cases(args.cases),
            variant=args.variant,
            backend=args.backend,
            rrf_k=args.rrf_k,
            warm_repeats=args.warm_repeats,
        )
        return_code = 0
    except DenseUnavailableError:
        report = {
            "status": "unavailable",
            "metadata": {
                "created_at": datetime.now(UTC).isoformat(),
                "variant": args.variant,
                "backend": args.backend,
            },
        }
        return_code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
