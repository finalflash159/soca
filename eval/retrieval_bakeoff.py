from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from eval.eval_hybrid_retrieval import build_embedding_model
from eval.retrieval_benchmark_data import (
    RetrievalDataset,
    load_beir_parquet,
    production_chunks,
)
from soca.knowledge.index.models import MarkdownChunk
from soca.knowledge.markdown_vault import SearchScoringConfig, tokenize_terms
from soca.knowledge.retriever import RankedHit
from soca.knowledge.retrievers.dense import DenseIndex, DenseRetriever, EmbeddingModel
from soca.knowledge.retrievers.rrf import reciprocal_rank_fusion
from soca.knowledge.retrievers.sparse_chunk import SparseChunkRetriever

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "benchmarks" / "retrieval" / "public"
REGISTERED_DATASETS = {
    "tvpl": ("public_screening", None, False),
    "arguana-vn": (
        "public_screening",
        "test-00000-of-00001.parquet",
        True,
    ),
    "scifact-vn": ("public_screening", "test-00000-of-00001.parquet", False),
}


class Ranker(Protocol):
    @property
    def artifact_bytes(self) -> int: ...

    def rank(self, query: str, *, limit: int) -> list[RankedHit]: ...


@dataclass(frozen=True)
class QueryMeasurement:
    query_id: str
    recall_at_5: float
    reciprocal_rank_at_10: float
    ndcg_at_10: float
    precision_at_3: float
    latency_ms: float
    retrieved_documents: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    sparse: str | None
    dense: str | None


class CustomSparseRanker:
    def __init__(self, chunks: tuple[MarkdownChunk, ...]) -> None:
        self._ranker = SparseChunkRetriever(chunks, SearchScoringConfig())

    @property
    def artifact_bytes(self) -> int:
        return 0

    def rank(self, query: str, *, limit: int) -> list[RankedHit]:
        return self._ranker.rank(query, limit=limit)


class Bm25Ranker:
    def __init__(self, chunks: tuple[MarkdownChunk, ...]) -> None:
        import bm25s

        self._chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
        tokenized = [list(tokenize_terms(chunk.text)) for chunk in chunks]
        self._retriever = bm25s.BM25(method="lucene")
        self._retriever.index(tokenized, show_progress=False)
        scores = self._retriever.scores
        self._artifact_bytes = sum(
            value.nbytes
            for value in scores.values()
            if isinstance(value, np.ndarray)
        )

    @property
    def artifact_bytes(self) -> int:
        return self._artifact_bytes

    def rank(self, query: str, *, limit: int) -> list[RankedHit]:
        terms = list(tokenize_terms(query))
        if not terms:
            return []
        result = self._retriever.retrieve(
            [terms],
            k=min(limit, len(self._chunk_ids)),
            show_progress=False,
        )
        indices = np.asarray(result.documents[0], dtype=np.int64)
        scores = np.asarray(result.scores[0], dtype=np.float32)
        return [
            RankedHit(
                chunk_id=self._chunk_ids[int(index)],
                rank=rank,
                score=float(score),
            )
            for rank, (index, score) in enumerate(
                zip(indices, scores, strict=True),
                start=1,
            )
        ]


class ExactDenseRanker:
    def __init__(
        self,
        chunks: tuple[MarkdownChunk, ...],
        model: EmbeddingModel,
        *,
        batch_size: int,
    ) -> None:
        vectors: list[np.ndarray] = []
        total_batches = math.ceil(len(chunks) / batch_size)
        for batch_number, start in enumerate(range(0, len(chunks), batch_size), start=1):
            texts = tuple(chunk.text for chunk in chunks[start : start + batch_size])
            vectors.append(model.embed_documents(texts))
            if batch_number == total_batches or batch_number % 20 == 0:
                print(
                    f"  embedded {min(start + batch_size, len(chunks))}/{len(chunks)} chunks",
                    flush=True,
                )
        matrix = np.vstack(vectors)
        digest = hashlib.sha256()
        for chunk in chunks:
            digest.update(chunk.chunk_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(chunk.text.encode("utf-8"))
            digest.update(b"\0")
        self._ranker = DenseRetriever(
            DenseIndex(
                model_id=model.model_id,
                source_digest=digest.hexdigest(),
                chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
                vectors=matrix,
            ),
            model,
        )
        self._artifact_bytes = matrix.nbytes

    @property
    def artifact_bytes(self) -> int:
        return self._artifact_bytes

    def rank(self, query: str, *, limit: int) -> list[RankedHit]:
        return self._ranker.rank(query, limit=limit)


class HybridRanker:
    def __init__(self, sparse: Ranker, dense: Ranker, *, rrf_k: int = 60) -> None:
        self._sparse = sparse
        self._dense = dense
        self._rrf_k = rrf_k

    @property
    def artifact_bytes(self) -> int:
        return self._sparse.artifact_bytes + self._dense.artifact_bytes

    def rank(self, query: str, *, limit: int) -> list[RankedHit]:
        candidate_limit = max(50, limit)
        fused = reciprocal_rank_fusion(
            (
                self._sparse.rank(query, limit=candidate_limit),
                self._dense.rank(query, limit=candidate_limit),
            ),
            k=self._rrf_k,
        )
        return [
            RankedHit(chunk_id=chunk_id, rank=rank, score=score)
            for rank, (chunk_id, score) in enumerate(fused[:limit], start=1)
        ]


def parse_candidate(value: str) -> CandidateSpec:
    parts = value.split(":")
    if parts == ["lexical_custom"]:
        return CandidateSpec(value, "lexical_custom", None)
    if parts == ["bm25"]:
        return CandidateSpec(value, "bm25", None)
    if len(parts) == 2 and parts[0] == "dense" and parts[1]:
        return CandidateSpec(value, None, parts[1])
    if (
        len(parts) == 3
        and parts[0] == "hybrid"
        and parts[1] in {"lexical_custom", "bm25"}
        and parts[2]
    ):
        return CandidateSpec(value, parts[1], parts[2])
    raise ValueError(f"invalid retrieval candidate: {value!r}")


def select_query_ids(
    qrels: dict[str, dict[str, int]],
    *,
    limit: int | None,
    seed: int,
) -> tuple[str, ...]:
    judged = tuple(
        sorted(
            qrels,
            key=lambda query_id: hashlib.sha256(
                f"{seed}\0{query_id}".encode()
            ).digest(),
        )
    )
    if limit is None:
        return judged
    if limit < 1:
        raise ValueError("query limit must be positive")
    return judged[:limit]


def ndcg_at_k(
    retrieved: Sequence[str],
    judgments: dict[str, int],
    *,
    k: int,
) -> float:
    gains = [
        (2 ** judgments.get(document_id, 0) - 1) / math.log2(rank + 1)
        for rank, document_id in enumerate(retrieved[:k], start=1)
    ]
    ideal_scores = sorted(judgments.values(), reverse=True)[:k]
    ideal = sum(
        (2**score - 1) / math.log2(rank + 1)
        for rank, score in enumerate(ideal_scores, start=1)
    )
    return sum(gains) / ideal if ideal else 0.0


def _measure_query(
    query_id: str,
    retrieved: Sequence[str],
    judgments: dict[str, int],
    *,
    latency_ms: float,
) -> QueryMeasurement:
    relevant = {document_id for document_id, score in judgments.items() if score > 0}
    recall = len(set(retrieved[:5]) & relevant) / len(relevant)
    reciprocal_rank = 0.0
    for rank, document_id in enumerate(retrieved[:10], start=1):
        if document_id in relevant:
            reciprocal_rank = 1.0 / rank
            break
    precision = len(set(retrieved[:3]) & relevant) / 3
    return QueryMeasurement(
        query_id,
        recall,
        reciprocal_rank,
        ndcg_at_k(retrieved, judgments, k=10),
        precision,
        latency_ms,
        tuple(retrieved[:10]),
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def summarize_measurements(
    measurements: Sequence[QueryMeasurement],
) -> dict[str, float]:
    if not measurements:
        raise ValueError("retrieval measurements must not be empty")
    latencies = [sample.latency_ms for sample in measurements]
    return {
        "recall_at_5": statistics.fmean(
            sample.recall_at_5 for sample in measurements
        ),
        "mrr_at_10": statistics.fmean(
            sample.reciprocal_rank_at_10 for sample in measurements
        ),
        "ndcg_at_10": statistics.fmean(
            sample.ndcg_at_10 for sample in measurements
        ),
        "precision_at_3": statistics.fmean(
            sample.precision_at_3 for sample in measurements
        ),
        "latency_mean_ms": statistics.fmean(latencies),
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "latency_p99_ms": _percentile(latencies, 0.99),
    }


def _build_ranker(
    spec: CandidateSpec,
    chunks: tuple[MarkdownChunk, ...],
    *,
    batch_size: int,
) -> Ranker:
    sparse: Ranker | None = None
    if spec.sparse == "lexical_custom":
        sparse = CustomSparseRanker(chunks)
    elif spec.sparse == "bm25":
        sparse = Bm25Ranker(chunks)
    if spec.dense is None:
        assert sparse is not None
        return sparse
    model = build_embedding_model(spec.dense)
    dense = ExactDenseRanker(chunks, model, batch_size=batch_size)
    return dense if sparse is None else HybridRanker(sparse, dense)


def evaluate_candidate(
    dataset: RetrievalDataset,
    spec: CandidateSpec,
    *,
    query_limit: int | None,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    chunks, document_paths = production_chunks(dataset)
    path_documents = {path: document_id for document_id, path in document_paths.items()}
    chunk_documents = {
        chunk.chunk_id: path_documents[chunk.document_path]
        for chunk in chunks
    }
    print(
        f"[{dataset.name}] build {spec.name}: "
        f"{len(dataset.documents)} docs / {len(chunks)} chunks",
        flush=True,
    )
    started = time.perf_counter()
    ranker = _build_ranker(spec, chunks, batch_size=batch_size)
    build_seconds = time.perf_counter() - started
    query_ids = select_query_ids(dataset.qrels, limit=query_limit, seed=seed)
    measurements: list[QueryMeasurement] = []
    for position, query_id in enumerate(query_ids, start=1):
        query_started = time.perf_counter_ns()
        chunk_hits = ranker.rank(dataset.queries[query_id], limit=50)
        latency_ms = (time.perf_counter_ns() - query_started) / 1_000_000
        documents: list[str] = []
        seen: set[str] = set()
        for hit in chunk_hits:
            document_id = chunk_documents[hit.chunk_id]
            if document_id not in seen:
                seen.add(document_id)
                documents.append(document_id)
        measurements.append(
            _measure_query(
                query_id,
                documents,
                dataset.qrels[query_id],
                latency_ms=latency_ms,
            )
        )
        if position == len(query_ids) or position % 100 == 0:
            print(f"  evaluated {position}/{len(query_ids)} queries", flush=True)
    return {
        "dataset": dataset.name,
        "dataset_class": dataset.dataset_class,
        "candidate": spec.name,
        "document_count": len(dataset.documents),
        "chunk_count": len(chunks),
        "query_count": len(query_ids),
        "excluded_upstream_qrels": len(dataset.excluded_qrels),
        "build_seconds": build_seconds,
        "artifact_bytes": ranker.artifact_bytes,
        "metrics": summarize_measurements(measurements),
        "measurements": [asdict(measurement) for measurement in measurements],
    }


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _output_path(argument: Path | None) -> Path:
    if argument is not None:
        return argument
    run_dir = os.environ.get("SOCA_BENCHMARK_RUN_DIR")
    if not run_dir:
        raise ValueError("--output or SOCA_BENCHMARK_RUN_DIR is required")
    return Path(run_dir) / "report.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bake off SoCa retrieval candidates on pinned non-demo data."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=tuple(REGISTERED_DATASETS),
        required=True,
    )
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--seed", type=int, default=20_260_729)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    specs = tuple(parse_candidate(value) for value in args.candidate)
    datasets = tuple(
        load_beir_parquet(
            args.data_root / name,
            name=name,
            dataset_class=REGISTERED_DATASETS[name][0],
            corpus_file=REGISTERED_DATASETS[name][1],
            allow_incomplete_qrels=REGISTERED_DATASETS[name][2],
        )
        for name in args.dataset
    )
    results = [
        evaluate_candidate(
            dataset,
            spec,
            query_limit=args.query_limit,
            seed=args.seed,
            batch_size=args.batch_size,
        )
        for dataset in datasets
        for spec in specs
    ]
    report = {
        "schema_version": 1,
        "suite": "retrieval-production-bakeoff",
        "dataset_policy": {
            "allowed_classes": [
                "public_screening",
                "sanitized_benchmark",
                "private_release",
            ],
            "observed_classes": sorted({result["dataset_class"] for result in results}),
            "demo_smoke_present": False,
        },
        "config": {
            "query_limit": args.query_limit,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "candidates": [spec.name for spec in specs],
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: _package_version(name)
                for name in (
                    "bm25s",
                    "faiss-cpu",
                    "fastembed",
                    "model2vec",
                    "numpy",
                    "sentence-transformers",
                    "usearch",
                )
            },
        },
        "results": results,
    }
    output = _output_path(args.output)
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = output.with_suffix(f"{output.suffix}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(output)
    print(f"report: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
