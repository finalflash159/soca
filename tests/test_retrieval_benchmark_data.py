from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from eval.retrieval_benchmark_data import (
    RetrievalDataset,
    load_beir_parquet,
    production_chunks,
)
from soca.knowledge.index.models import MarkdownChunk


def _write_parquet_dataset(root: Path) -> None:
    (root / "corpus").mkdir(parents=True)
    (root / "queries").mkdir()
    (root / "qrels").mkdir()
    pd.DataFrame(
        [
            {"_id": "doc/one", "title": "Bayes", "text": "Xác suất có điều kiện."},
            {"_id": "doc-two", "title": "", "text": "ONNX Runtime chạy mô hình."},
        ]
    ).to_parquet(root / "corpus" / "part.parquet")
    pd.DataFrame(
        [{"_id": "q1", "text": "Bayes là gì?"}]
    ).to_parquet(root / "queries" / "part.parquet")
    pd.DataFrame(
        [{"query-id": "q1", "corpus-id": "doc/one", "score": 2}]
    ).to_parquet(root / "qrels" / "part.parquet")


def test_load_beir_parquet_normalizes_ids_and_qrels(tmp_path: Path) -> None:
    _write_parquet_dataset(tmp_path)

    dataset = load_beir_parquet(
        tmp_path,
        name="public-test",
        dataset_class="public_screening",
    )

    assert dataset.documents["doc/one"].title == "Bayes"
    assert dataset.queries == {"q1": "Bayes là gì?"}
    assert dataset.qrels == {"q1": {"doc/one": 2}}


def test_quality_dataset_rejects_demo_class() -> None:
    with pytest.raises(ValueError, match="not eligible"):
        RetrievalDataset(
            name="bad",
            dataset_class="demo_smoke",
            documents={},
            queries={},
            qrels={},
        )


def test_production_chunks_use_safe_paths_and_preserve_document_mapping(
    tmp_path: Path,
) -> None:
    _write_parquet_dataset(tmp_path)
    dataset = load_beir_parquet(
        tmp_path,
        name="public-test",
        dataset_class="public_screening",
    )

    chunks, document_paths = production_chunks(dataset)

    assert chunks
    assert all(isinstance(chunk, MarkdownChunk) for chunk in chunks)
    assert document_paths["doc/one"].startswith("wiki/benchmark/public-test/")
    assert document_paths["doc/one"].endswith(".md")
    assert all(chunk.document_path in document_paths.values() for chunk in chunks)


def test_load_beir_parquet_rejects_qrels_for_unknown_documents(tmp_path: Path) -> None:
    _write_parquet_dataset(tmp_path)
    qrels_path = tmp_path / "qrels" / "part.parquet"
    pd.DataFrame(
        [{"query-id": "q1", "corpus-id": "missing", "score": 1}]
    ).to_parquet(qrels_path)

    with pytest.raises(ValueError, match="unknown corpus"):
        load_beir_parquet(
            tmp_path,
            name="public-test",
            dataset_class="public_screening",
        )


def test_incomplete_public_screening_can_record_and_exclude_invalid_qrels(
    tmp_path: Path,
) -> None:
    _write_parquet_dataset(tmp_path)
    qrels_path = tmp_path / "qrels" / "part.parquet"
    pd.DataFrame(
        [
            {"query-id": "q1", "corpus-id": "doc/one", "score": 2},
            {"query-id": "q1", "corpus-id": "missing", "score": 1},
        ]
    ).to_parquet(qrels_path)

    dataset = load_beir_parquet(
        tmp_path,
        name="upstream-incomplete",
        dataset_class="public_screening",
        allow_incomplete_qrels=True,
    )

    assert dataset.qrels == {"q1": {"doc/one": 2}}
    assert dataset.excluded_qrels == (("q1", "missing"),)
