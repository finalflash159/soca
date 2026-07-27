from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from soca.knowledge.retrievers.dense import DenseIndex, DenseRetriever


class FakeEmbeddingModel:
    def __init__(
        self,
        query_vector: np.ndarray,
        *,
        model_id: str = "fake:model",
    ) -> None:
        self._model_id = model_id
        self.query_vector = query_vector
        self.queries: list[str] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    def embed_documents(self, texts: tuple[str, ...]) -> np.ndarray:
        raise AssertionError(f"dense retrieval must not embed documents: {texts!r}")

    def embed_query(self, text: str) -> np.ndarray:
        self.queries.append(text)
        return self.query_vector


def _index(
    vectors: np.ndarray,
    *,
    chunk_ids: tuple[str, ...] = ("chunk-b", "chunk-a"),
    model_id: str = "fake:model",
) -> DenseIndex:
    return DenseIndex(
        model_id=model_id,
        source_digest="a" * 64,
        chunk_ids=chunk_ids,
        vectors=vectors,
    )


def test_dense_index_normalizes_and_freezes_vectors() -> None:
    source = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float64)

    index = _index(source)

    assert index.vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(index.vectors, axis=1), np.ones(2))
    assert not index.vectors.flags.writeable
    with pytest.raises(ValueError):
        index.vectors[0, 0] = 99.0
    with pytest.raises(FrozenInstanceError):
        index.model_id = "different"  # type: ignore[misc]


def test_dense_index_copies_chunk_ids_into_an_immutable_tuple() -> None:
    chunk_ids = ["chunk-a"]

    index = DenseIndex(
        model_id="fake:model",
        source_digest="a" * 64,
        chunk_ids=chunk_ids,  # type: ignore[arg-type]
        vectors=np.ones((1, 2), dtype=np.float32),
    )
    chunk_ids.append("chunk-b")

    assert index.chunk_ids == ("chunk-a",)


@pytest.mark.parametrize(
    ("vectors", "chunk_ids"),
    [
        (np.array([1.0, 2.0], dtype=np.float32), ("chunk-a",)),
        (np.empty((1, 0), dtype=np.float32), ("chunk-a",)),
        (np.array([[1.0, 2.0]], dtype=np.float32), ("chunk-a", "chunk-b")),
        (np.array([[0.0, 0.0]], dtype=np.float32), ("chunk-a",)),
        (np.array([[np.nan, 1.0]], dtype=np.float32), ("chunk-a",)),
        (np.array([[np.inf, 1.0]], dtype=np.float32), ("chunk-a",)),
    ],
)
def test_dense_index_rejects_invalid_embedding_matrices(
    vectors: np.ndarray,
    chunk_ids: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        _index(vectors, chunk_ids=chunk_ids)


@pytest.mark.parametrize(
    ("model_id", "source_digest", "chunk_ids"),
    [
        ("", "a" * 64, ("chunk-a",)),
        (" ", "a" * 64, ("chunk-a",)),
        ("fake:model", "", ("chunk-a",)),
        ("fake:model", "a" * 63, ("chunk-a",)),
        ("fake:model", "A" * 64, ("chunk-a",)),
        ("fake:model", "../unsafe", ("chunk-a",)),
        ("fake:model", "a" * 64, ("",)),
        ("fake:model", "a" * 64, (" ",)),
        ("fake:model", "a" * 64, ("chunk-a", "chunk-a")),
    ],
)
def test_dense_index_rejects_invalid_identifiers(
    model_id: str,
    source_digest: str,
    chunk_ids: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        DenseIndex(
            model_id=model_id,
            source_digest=source_digest,
            chunk_ids=chunk_ids,
            vectors=np.ones((len(chunk_ids), 2), dtype=np.float32),
        )


def test_dense_retriever_ranks_by_cosine_then_chunk_id_for_stable_ties() -> None:
    index = _index(
        np.array(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        chunk_ids=("chunk-z", "chunk-a", "chunk-m"),
    )
    model = FakeEmbeddingModel(np.array([3.0, 0.0], dtype=np.float32))
    retriever = DenseRetriever(index, model)

    ranking = retriever.rank_with_score("question", limit=2)

    assert [hit.chunk_id for hit in ranking.hits] == ["chunk-a", "chunk-z"]
    assert [hit.rank for hit in ranking.hits] == [1, 2]
    np.testing.assert_allclose([hit.score for hit in ranking.hits], [1.0, 1.0])
    assert ranking.max_score == pytest.approx(1.0)
    assert model.queries == ["question"]
    assert retriever.rank("question", limit=1)[0].chunk_id == "chunk-a"


def test_dense_retriever_rejects_a_model_mismatch() -> None:
    index = _index(np.eye(2, dtype=np.float32))
    model = FakeEmbeddingModel(
        np.array([1.0, 0.0], dtype=np.float32),
        model_id="different:model",
    )

    with pytest.raises(ValueError, match="model"):
        DenseRetriever(index, model)


def test_dense_retriever_rejects_non_positive_limit_before_embedding() -> None:
    model = FakeEmbeddingModel(np.array([1.0, 0.0], dtype=np.float32))
    retriever = DenseRetriever(_index(np.eye(2, dtype=np.float32)), model)

    with pytest.raises(ValueError, match="positive"):
        retriever.rank("question", limit=0)

    assert model.queries == []


def test_dense_retriever_skips_empty_query_and_empty_index() -> None:
    model = FakeEmbeddingModel(np.array([1.0, 0.0], dtype=np.float32))
    retriever = DenseRetriever(_index(np.eye(2, dtype=np.float32)), model)
    empty_retriever = DenseRetriever(
        _index(
            np.empty((0, 2), dtype=np.float32),
            chunk_ids=(),
        ),
        model,
    )

    assert retriever.rank(" \n ", limit=2) == []
    assert empty_retriever.rank("question", limit=2) == []
    assert model.queries == []
    assert not empty_retriever.available


@pytest.mark.parametrize(
    "query_vector",
    [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([np.nan, 0.0], dtype=np.float32),
        np.array([0.0, 0.0], dtype=np.float32),
    ],
)
def test_dense_retriever_rejects_invalid_query_vectors(
    query_vector: np.ndarray,
) -> None:
    retriever = DenseRetriever(
        _index(np.eye(2, dtype=np.float32)),
        FakeEmbeddingModel(query_vector),
    )

    with pytest.raises(ValueError):
        retriever.rank("question", limit=2)
