from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

import numpy as np

from eval.eval_hybrid_retrieval import build_embedding_model
from eval.retrieval_benchmark_data import (
    RetrievalDataset,
    load_beir_parquet,
    load_vire_csv,
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
VIRE_DATASETS = {
    "vire-viquad2": ("UIT-ViQuAD2.csv", "id"),
    "vire-vinewsqa": ("ViNewsQA.csv", "id"),
    "vire-educoqa": ("EduCoQA.csv", "qid"),
    "vire-vimedaqa": ("ViMedAQA_v2.csv", "idx"),
    "vire-alqac": ("ALQAC.csv", None),
    "vire-csconda": ("CSConDa.csv", None),
}
T = TypeVar("T")


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
    fusion: Literal["rrf", "linear"] | None = None
    dense_weight: float = 0.5
    reranker: str | None = None
    rerank_top_k: int = 0


@dataclass(frozen=True)
class BuiltRanker:
    ranker: Ranker
    cold_build_seconds: float
    reused_components: tuple[str, ...]


class StaticRanker:
    def __init__(self, hits: tuple[RankedHit, ...]) -> None:
        self._hits = hits

    @property
    def artifact_bytes(self) -> int:
        return 0

    def rank(self, query: str, *, limit: int) -> list[RankedHit]:
        del query
        return list(self._hits[:limit])


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


def _min_max_scores(hits: Sequence[RankedHit]) -> dict[str, float]:
    if not hits:
        return {}
    values = [hit.score for hit in hits]
    low = min(values)
    span = max(values) - low
    if span <= 1e-12:
        return {hit.chunk_id: 0.0 for hit in hits}
    return {hit.chunk_id: (hit.score - low) / span for hit in hits}


class LinearHybridRanker:
    def __init__(
        self,
        sparse: Ranker,
        dense: Ranker,
        *,
        dense_weight: float,
    ) -> None:
        if not 0 <= dense_weight <= 1:
            raise ValueError("dense weight must be in [0, 1]")
        self._sparse = sparse
        self._dense = dense
        self._dense_weight = dense_weight

    @property
    def artifact_bytes(self) -> int:
        return self._sparse.artifact_bytes + self._dense.artifact_bytes

    def rank(self, query: str, *, limit: int) -> list[RankedHit]:
        candidate_limit = max(50, limit)
        sparse = _min_max_scores(self._sparse.rank(query, limit=candidate_limit))
        dense = _min_max_scores(self._dense.rank(query, limit=candidate_limit))
        identifiers = set(sparse) | set(dense)
        scored = sorted(
            (
                (
                    identifier,
                    (1 - self._dense_weight) * sparse.get(identifier, 0.0)
                    + self._dense_weight * dense.get(identifier, 0.0),
                )
                for identifier in identifiers
            ),
            key=lambda item: (-item[1], item[0]),
        )
        return [
            RankedHit(chunk_id, rank, score)
            for rank, (chunk_id, score) in enumerate(scored[:limit], start=1)
        ]


class CrossEncoderScorer:
    def __init__(self, candidate: str, *, batch_size: int) -> None:
        model_root = (
            Path.home() / ".local" / "share" / "soca" / "models" / "eval" / candidate
        )
        if not model_root.is_dir():
            raise FileNotFoundError(f"reranker is not provisioned at {model_root}")
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(
            str(model_root),
            device="cpu",
            local_files_only=True,
            trust_remote_code=candidate == "rerank_gte_multilingual",
        )
        self._batch_size = batch_size
        self.artifact_bytes = sum(
            path.stat().st_size
            for path in model_root.rglob("*")
            if path.is_file() and ".git" not in path.parts
        )

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        values = self._model.predict(
            [[query, document] for document in documents],
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        scores = np.asarray(values, dtype=np.float32).reshape(-1)
        if scores.shape != (len(documents),) or not np.isfinite(scores).all():
            raise ValueError("reranker returned invalid scores")
        return scores


class RerankRanker:
    def __init__(
        self,
        base: Ranker,
        scorer: CrossEncoderScorer,
        chunks: tuple[MarkdownChunk, ...],
        *,
        top_k: int,
    ) -> None:
        if top_k < 1:
            raise ValueError("rerank top-k must be positive")
        self._base = base
        self._scorer = scorer
        self._texts = {chunk.chunk_id: chunk.text for chunk in chunks}
        self._top_k = top_k

    @property
    def artifact_bytes(self) -> int:
        return self._base.artifact_bytes + self._scorer.artifact_bytes

    def rank(self, query: str, *, limit: int) -> list[RankedHit]:
        candidates = self._base.rank(query, limit=max(limit, self._top_k))
        rerankable = candidates[: self._top_k]
        scores = self._scorer.score(
            query,
            [self._texts[hit.chunk_id] for hit in rerankable],
        )
        scored = sorted(
            zip(rerankable, scores, strict=True),
            key=lambda item: (-float(item[1]), item[0].chunk_id),
        )
        tail = candidates[self._top_k :]
        ordered = [item[0] for item in scored] + tail
        score_by_id = {item.chunk_id: float(score) for item, score in scored}
        return [
            RankedHit(
                item.chunk_id,
                rank,
                score_by_id.get(item.chunk_id, item.score),
            )
            for rank, item in enumerate(ordered[:limit], start=1)
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
        return CandidateSpec(value, parts[1], parts[2], fusion="rrf")
    if (
        len(parts) == 3
        and parts[0] == "hybrid_rrf"
        and parts[1] in {"lexical_custom", "bm25"}
        and parts[2]
    ):
        return CandidateSpec(value, parts[1], parts[2], fusion="rrf")
    if (
        len(parts) == 4
        and parts[0] == "hybrid_linear"
        and parts[1] in {"lexical_custom", "bm25"}
        and parts[2]
    ):
        try:
            dense_weight = float(parts[3])
        except ValueError as exc:
            raise ValueError(f"invalid dense fusion weight: {parts[3]!r}") from exc
        if not 0 <= dense_weight <= 1:
            raise ValueError("dense fusion weight must be in [0, 1]")
        return CandidateSpec(
            value,
            parts[1],
            parts[2],
            fusion="linear",
            dense_weight=dense_weight,
        )
    if (
        len(parts) == 7
        and parts[0] == "rerank"
        and parts[1] in {"hybrid_rrf", "hybrid_linear"}
        and parts[2] in {"lexical_custom", "bm25"}
        and parts[3]
        and parts[4]
    ):
        if parts[1] == "hybrid_rrf":
            if parts[5]:
                raise ValueError("RRF reranker candidate must leave weight empty")
            dense_weight = 0.5
        else:
            try:
                dense_weight = float(parts[5])
            except ValueError as exc:
                raise ValueError(f"invalid dense fusion weight: {parts[5]!r}") from exc
            if not 0 <= dense_weight <= 1:
                raise ValueError("dense fusion weight must be in [0, 1]")
        try:
            top_k = int(parts[6])
        except ValueError as exc:
            raise ValueError(f"invalid rerank top-k: {parts[6]!r}") from exc
        if top_k < 1:
            raise ValueError("rerank top-k must be positive")
        return CandidateSpec(
            value,
            parts[2],
            parts[3],
            fusion="rrf" if parts[1] == "hybrid_rrf" else "linear",
            dense_weight=dense_weight,
            reranker=parts[4],
            rerank_top_k=top_k,
        )
    if (
        len(parts) == 6
        and parts[0] == "rerank"
        and parts[1] == "hybrid_rrf"
        and parts[2] in {"lexical_custom", "bm25"}
        and parts[3]
        and parts[4]
    ):
        try:
            top_k = int(parts[5])
        except ValueError as exc:
            raise ValueError(f"invalid rerank top-k: {parts[5]!r}") from exc
        if top_k < 1:
            raise ValueError("rerank top-k must be positive")
        return CandidateSpec(
            value,
            parts[2],
            parts[3],
            fusion="rrf",
            reranker=parts[4],
            rerank_top_k=top_k,
        )
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


class RankerFactory:
    def __init__(
        self,
        chunks: tuple[MarkdownChunk, ...],
        *,
        batch_size: int,
    ) -> None:
        self._chunks = chunks
        self._batch_size = batch_size
        self._sparse: dict[str, tuple[Ranker, float]] = {}
        self._dense: dict[str, tuple[Ranker, float]] = {}
        self._rerankers: dict[str, tuple[CrossEncoderScorer, float]] = {}

    @staticmethod
    def _timed(factory: Callable[[], T]) -> tuple[T, float]:
        started = time.perf_counter()
        value = factory()
        return value, time.perf_counter() - started

    def _sparse_ranker(self, name: str) -> tuple[Ranker, float, bool]:
        cached = self._sparse.get(name)
        if cached is not None:
            return cached[0], cached[1], True
        if name == "lexical_custom":
            built = self._timed(lambda: CustomSparseRanker(self._chunks))
        elif name == "bm25":
            built = self._timed(lambda: Bm25Ranker(self._chunks))
        else:
            raise ValueError(f"unknown sparse candidate: {name}")
        self._sparse[name] = built
        return built[0], built[1], False

    def _dense_ranker(self, name: str) -> tuple[Ranker, float, bool]:
        cached = self._dense.get(name)
        if cached is not None:
            return cached[0], cached[1], True

        def build() -> Ranker:
            model = build_embedding_model(name)
            return ExactDenseRanker(
                self._chunks,
                model,
                batch_size=self._batch_size,
            )

        built = self._timed(build)
        self._dense[name] = built
        return built[0], built[1], False

    def _reranker(self, name: str) -> tuple[CrossEncoderScorer, float, bool]:
        cached = self._rerankers.get(name)
        if cached is not None:
            return cached[0], cached[1], True
        built = self._timed(
            lambda: CrossEncoderScorer(name, batch_size=self._batch_size)
        )
        self._rerankers[name] = built
        return built[0], built[1], False

    def build(self, spec: CandidateSpec) -> BuiltRanker:
        cold_seconds = 0.0
        reused: list[str] = []
        sparse: Ranker | None = None
        dense: Ranker | None = None
        if spec.sparse is not None:
            sparse, elapsed, cached = self._sparse_ranker(spec.sparse)
            cold_seconds += elapsed
            if cached:
                reused.append(f"sparse:{spec.sparse}")
        if spec.dense is not None:
            dense, elapsed, cached = self._dense_ranker(spec.dense)
            cold_seconds += elapsed
            if cached:
                reused.append(f"dense:{spec.dense}")

        if sparse is None:
            assert dense is not None
            ranker = dense
        elif dense is None:
            ranker = sparse
        elif spec.fusion == "linear":
            ranker = LinearHybridRanker(
                sparse,
                dense,
                dense_weight=spec.dense_weight,
            )
        else:
            ranker = HybridRanker(sparse, dense)

        if spec.reranker is not None:
            scorer, elapsed, cached = self._reranker(spec.reranker)
            cold_seconds += elapsed
            if cached:
                reused.append(f"reranker:{spec.reranker}")
            ranker = RerankRanker(
                ranker,
                scorer,
                self._chunks,
                top_k=spec.rerank_top_k,
            )
        return BuiltRanker(ranker, cold_seconds, tuple(reused))


def evaluate_candidate(
    dataset: RetrievalDataset,
    spec: CandidateSpec,
    *,
    chunks: tuple[MarkdownChunk, ...],
    document_paths: dict[str, str],
    ranker_factory: RankerFactory,
    query_limit: int | None,
    seed: int,
) -> dict[str, Any]:
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
    built = ranker_factory.build(spec)
    build_seconds = time.perf_counter() - started
    ranker = built.ranker
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
        "status": "ok",
        "dataset": dataset.name,
        "dataset_class": dataset.dataset_class,
        "candidate": spec.name,
        "document_count": len(dataset.documents),
        "chunk_count": len(chunks),
        "query_count": len(query_ids),
        "excluded_upstream_qrels": len(dataset.excluded_qrels),
        "build_seconds": build_seconds,
        "cold_component_build_seconds": built.cold_build_seconds,
        "reused_components": list(built.reused_components),
        "artifact_bytes": ranker.artifact_bytes,
        "metrics": summarize_measurements(measurements),
        "measurements": [asdict(measurement) for measurement in measurements],
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _load_dataset(name: str, data_root: Path) -> RetrievalDataset:
    vire = VIRE_DATASETS.get(name)
    if vire is not None:
        filename, query_id_column = vire
        return load_vire_csv(
            data_root / "vire" / "data" / filename,
            name=name,
            dataset_class="public_screening",
            query_id_column=query_id_column,
        )
    dataset_class, corpus_file, allow_incomplete = REGISTERED_DATASETS[name]
    return load_beir_parquet(
        data_root / name,
        name=name,
        dataset_class=dataset_class,
        corpus_file=corpus_file,
        allow_incomplete_qrels=allow_incomplete,
    )


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
        choices=tuple([*REGISTERED_DATASETS, *VIRE_DATASETS]),
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
    datasets = tuple(_load_dataset(name, args.data_root) for name in args.dataset)
    results: list[dict[str, Any]] = []
    for dataset in datasets:
        chunks, document_paths = production_chunks(dataset)
        ranker_factory = RankerFactory(chunks, batch_size=args.batch_size)
        for spec in specs:
            try:
                result = evaluate_candidate(
                    dataset,
                    spec,
                    chunks=chunks,
                    document_paths=document_paths,
                    ranker_factory=ranker_factory,
                    query_limit=args.query_limit,
                    seed=args.seed,
                )
            except Exception as exc:
                print(
                    f"[{dataset.name}] {spec.name} failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                result = {
                    "status": "failed",
                    "dataset": dataset.name,
                    "dataset_class": dataset.dataset_class,
                    "candidate": spec.name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "peak_rss_bytes": _peak_rss_bytes(),
                }
            results.append(result)
    report = {
        "schema_version": 1,
        "suite": "retrieval-production-bakeoff",
        "dataset_policy": {
            "allowed_classes": [
                "public_screening",
                "sanitized_benchmark",
                "private_release",
            ],
            "observed_classes": sorted(
                {str(result["dataset_class"]) for result in results}
            ),
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
