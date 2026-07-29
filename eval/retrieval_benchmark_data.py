from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from eval.retrieval_sources import DatasetClass
from soca.knowledge.base import KnowledgeDocument
from soca.knowledge.index.chunker import chunk_markdown
from soca.knowledge.index.models import MarkdownChunk

DatasetClassValue = Literal[
    "public_screening",
    "sanitized_benchmark",
    "private_release",
]


@dataclass(frozen=True)
class BenchmarkDocument:
    document_id: str
    title: str
    text: str


@dataclass(frozen=True)
class RetrievalDataset:
    name: str
    dataset_class: DatasetClassValue | str
    documents: dict[str, BenchmarkDocument]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    excluded_qrels: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        try:
            resolved_class = DatasetClass(self.dataset_class)
        except ValueError as exc:
            raise ValueError(
                f"dataset class {self.dataset_class!r} is not eligible for quality"
            ) from exc
        object.__setattr__(self, "dataset_class", resolved_class.value)
        if not self.name.strip():
            raise ValueError("dataset name must not be empty")
        if not self.documents:
            raise ValueError("retrieval dataset requires documents")
        if not self.queries:
            raise ValueError("retrieval dataset requires queries")
        if not self.qrels:
            raise ValueError("retrieval dataset requires qrels")

        unknown_queries = set(self.qrels) - set(self.queries)
        if unknown_queries:
            raise ValueError(
                "qrels reference unknown queries: " + ", ".join(sorted(unknown_queries)[:5])
            )
        unknown_documents = {
            document_id
            for judgments in self.qrels.values()
            for document_id in judgments
            if document_id not in self.documents
        }
        if unknown_documents:
            raise ValueError(
                "qrels reference unknown corpus documents: "
                + ", ".join(sorted(unknown_documents)[:5])
            )


def _one_parquet(directory: Path, filename: str | None = None) -> Path:
    if filename is not None:
        candidate = directory / filename
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate
    candidates = tuple(sorted(directory.glob("*.parquet")))
    if len(candidates) != 1:
        raise ValueError(f"{directory} must contain exactly one parquet shard")
    return candidates[0]


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def load_beir_parquet(
    root: Path,
    *,
    name: str,
    dataset_class: DatasetClassValue,
    corpus_file: str | None = None,
    allow_incomplete_qrels: bool = False,
) -> RetrievalDataset:
    corpus_frame = pd.read_parquet(_one_parquet(root / "corpus", corpus_file))
    query_frame = pd.read_parquet(_one_parquet(root / "queries"))
    qrels_frame = pd.read_parquet(_one_parquet(root / "qrels"))
    corpus_id_column = "_id" if "_id" in corpus_frame.columns else "id"
    query_id_column = "_id" if "_id" in query_frame.columns else "id"
    required_corpus = {corpus_id_column, "title", "text"}
    required_queries = {query_id_column, "text"}
    required_qrels = {"query-id", "corpus-id", "score"}
    if not required_corpus.issubset(corpus_frame.columns):
        raise ValueError("corpus parquet does not use the expected BEIR schema")
    if not required_queries.issubset(query_frame.columns):
        raise ValueError("query parquet does not use the expected BEIR schema")
    if not required_qrels.issubset(qrels_frame.columns):
        raise ValueError("qrels parquet does not use the expected BEIR schema")

    documents: dict[str, BenchmarkDocument] = {}
    for row in corpus_frame.to_dict("records"):
        document_id = _string(row[corpus_id_column], field="corpus id")
        text = _string(row["text"], field=f"{document_id} text")
        title_value = row["title"]
        title = title_value.strip() if isinstance(title_value, str) else ""
        if document_id in documents:
            raise ValueError(f"duplicate corpus id: {document_id}")
        documents[document_id] = BenchmarkDocument(document_id, title, text)

    queries: dict[str, str] = {}
    for row in query_frame.to_dict("records"):
        query_id = _string(row[query_id_column], field="query id")
        query = _string(row["text"], field=f"{query_id} query")
        if query_id in queries:
            raise ValueError(f"duplicate query id: {query_id}")
        queries[query_id] = query

    qrels: dict[str, dict[str, int]] = {}
    excluded_qrels: list[tuple[str, str]] = []
    for row in qrels_frame.to_dict("records"):
        query_id = _string(row["query-id"], field="qrel query id")
        document_id = _string(row["corpus-id"], field="qrel corpus id")
        score_value = row["score"]
        if isinstance(score_value, bool) or not isinstance(score_value, int | float):
            raise ValueError("qrel score must be numeric")
        score = int(score_value)
        if score <= 0:
            continue
        if document_id not in documents:
            if allow_incomplete_qrels:
                excluded_qrels.append((query_id, document_id))
                continue
        judgments = qrels.setdefault(query_id, {})
        judgments[document_id] = max(score, judgments.get(document_id, 0))

    return RetrievalDataset(
        name,
        dataset_class,
        documents,
        queries,
        qrels,
        tuple(sorted(excluded_qrels)),
    )


def production_chunks(
    dataset: RetrievalDataset,
) -> tuple[tuple[MarkdownChunk, ...], dict[str, str]]:
    chunks: list[MarkdownChunk] = []
    document_paths: dict[str, str] = {}
    for document_id, item in sorted(dataset.documents.items()):
        digest = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
        path = f"wiki/benchmark/{dataset.name}/{digest}.md"
        title = item.title or document_id
        text = f"# {title}\n\n{item.text}"
        document = KnowledgeDocument(
            id=document_id,
            path=path,
            title=title,
            text=text,
        )
        document_paths[document_id] = path
        chunks.extend(chunk_markdown(document))
    return tuple(chunks), document_paths
