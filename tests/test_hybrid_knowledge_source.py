from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from soca.knowledge.hybrid_source import (
    DenseUnavailableError,
    HybridConfig,
    HybridKnowledgeSource,
    RetrievalDiagnostics,
)


class FakeEmbeddingModel:
    model_id = "fake:model"

    def __init__(self, *, fail_documents: bool = False, fail_queries: bool = False) -> None:
        self.fail_documents = fail_documents
        self.fail_queries = fail_queries
        self.document_calls: list[tuple[str, ...]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, texts: tuple[str, ...]) -> np.ndarray:
        self.document_calls.append(texts)
        if self.fail_documents:
            raise RuntimeError("fake dense construction failure")
        return np.array(
            [[float(index + 1), 1.0] for index, _ in enumerate(texts)],
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> np.ndarray:
        self.query_calls.append(text)
        if self.fail_queries:
            raise RuntimeError("fake dense query failure")
        return np.array([1.0, 0.0], dtype=np.float32)


def _make_vault(root: Path) -> None:
    wiki = root / "wiki"
    wiki.mkdir()
    (wiki / "nutrition.md").write_text(
        "# Nutrition\nProtein supports muscle.\n\n# Hydration\nDrink enough water.",
        encoding="utf-8",
    )
    (wiki / "bayes.md").write_text(
        "# Bayes\nBayes updates beliefs with evidence.",
        encoding="utf-8",
    )


def _source(
    root: Path,
    index_home: Path,
    *,
    model: FakeEmbeddingModel | None,
    config: HybridConfig,
) -> HybridKnowledgeSource:
    return HybridKnowledgeSource(
        root,
        model=model,
        index_home=index_home,
        include_globs=("wiki/**/*.md",),
        config=config,
    )


def test_sparse_only_mode_does_not_call_dense_backend(tmp_path: Path) -> None:
    _make_vault(tmp_path)
    model = FakeEmbeddingModel()
    source = _source(
        tmp_path,
        tmp_path / "index",
        model=model,
        config=HybridConfig(dense_enabled=False),
    )

    batch = source.retrieve("protein", limit=5)

    assert batch.hits
    assert all(hit.line_start is not None for hit in batch.hits)
    assert model.document_calls == []
    assert model.query_calls == []
    assert batch.diagnostics.sparse_state == "ready"
    assert batch.diagnostics.dense_state == "absent"


def test_hybrid_rejects_a_missing_dense_model_instead_of_using_sparse(
    tmp_path: Path,
) -> None:
    _make_vault(tmp_path)
    source = _source(
        tmp_path,
        tmp_path / "index",
        model=None,
        config=HybridConfig(
            sparse_backend="bm25",
            fusion="linear",
            dense_weight=0.75,
        ),
    )

    with pytest.raises(DenseUnavailableError, match="has no model"):
        source.retrieve("bayes", limit=5)


def test_hybrid_keeps_two_chunks_from_one_document_as_distinct_hits(tmp_path: Path) -> None:
    _make_vault(tmp_path)
    source = _source(
        tmp_path,
        tmp_path / "index",
        model=FakeEmbeddingModel(),
        config=HybridConfig(per_retriever_limit=10),
    )

    source.build_index()
    hits = source.search("nutrition", limit=10)

    assert hits
    nutrition_hits = [hit for hit in hits if hit.document.path == "wiki/nutrition.md"]
    assert len(nutrition_hits) >= 2
    assert len({hit.document.id for hit in nutrition_hits}) == len(nutrition_hits)
    assert all(hit.document.id.startswith("wiki/") for hit in nutrition_hits)
    assert all(hit.line_end >= hit.line_start >= 1 for hit in nutrition_hits)
    assert all(hit.retrieval_backend in {"hybrid", "dense", "lexical_custom"} for hit in hits)
    assert all(hit.fusion_score is not None for hit in hits)


def test_custom_lexical_fusion_aggregates_chunks_at_document_boundary(
    tmp_path: Path,
) -> None:
    _make_vault(tmp_path)
    source = _source(
        tmp_path,
        tmp_path / "index",
        model=FakeEmbeddingModel(),
        config=HybridConfig(
            sparse_backend="lexical_custom",
            dense_weight=0.25,
        ),
    )

    source.build_index()
    hits = source.search("protein", limit=10)

    paths = [hit.document.path for hit in hits]
    assert len(paths) == len(set(paths))
    assert all(hit.document.id in paths for hit in hits)
    assert all(hit.fusion_score is not None for hit in hits)


def test_custom_lexical_fusion_preserves_the_query_local_passage(
    tmp_path: Path,
) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "weekly.md").write_text(
        "\n".join(
            [
                "# Weekly review",
                "",
                "## Unfinished",
                "unfinished needle receipt remains open",
                "",
                "## Later section",
                "future scheduling details without the requested item",
            ]
        ),
        encoding="utf-8",
    )
    source = _source(
        tmp_path,
        tmp_path / "index",
        model=FakeEmbeddingModel(),
        config=HybridConfig(
            sparse_backend="lexical_custom",
            dense_weight=0.25,
            per_retriever_limit=10,
        ),
    )

    source.build_index()
    hits = source.search("unfinished needle", limit=5)

    assert hits[0].document.path == "wiki/weekly.md"
    assert "unfinished needle" in hits[0].snippet


def test_dense_retrieval_is_not_blocked_by_a_lexical_miss(tmp_path: Path) -> None:
    _make_vault(tmp_path)
    model = FakeEmbeddingModel()
    source = _source(
        tmp_path,
        tmp_path / "index",
        model=model,
        config=HybridConfig(sparse_enabled=False, dense_enabled=True),
    )

    source.build_index()
    batch = source.retrieve("khái niệm không xuất hiện trong vault", limit=2)

    assert batch.hits
    assert model.query_calls == ["khái niệm không xuất hiện trong vault"]
    assert batch.diagnostics.dense_state == "ready"
    assert batch.diagnostics.sparse_state == "absent"


def test_dense_construction_failure_raises_in_hybrid_mode(
    tmp_path: Path,
) -> None:
    _make_vault(tmp_path)
    model = FakeEmbeddingModel(fail_documents=True)
    source = _source(
        tmp_path,
        tmp_path / "index",
        model=model,
        config=HybridConfig(),
    )

    with pytest.raises(DenseUnavailableError, match="index refresh failed"):
        source.build_index()


def test_dense_only_failure_raises_explicit_error(tmp_path: Path) -> None:
    _make_vault(tmp_path)
    source = _source(
        tmp_path,
        tmp_path / "index",
        model=FakeEmbeddingModel(fail_documents=True),
        config=HybridConfig(sparse_enabled=False),
    )

    with pytest.raises(DenseUnavailableError, match="index refresh failed"):
        source.build_index()


def test_dense_query_failure_raises_instead_of_using_sparse(tmp_path: Path) -> None:
    _make_vault(tmp_path)
    index_home = tmp_path / "index"
    source = _source(
        tmp_path,
        index_home,
        model=FakeEmbeddingModel(),
        config=HybridConfig(),
    )
    source.build_index()
    source._model = FakeEmbeddingModel(fail_queries=True)  # type: ignore[attr-defined]

    with pytest.raises(DenseUnavailableError, match="query failed"):
        source.retrieve("bayes", limit=5)


@pytest.mark.parametrize("dense_state", ("model_missing", "stale", "incompatible", "failed"))
def test_dense_only_unusable_states_are_unavailable(dense_state: str) -> None:
    diagnostics = RetrievalDiagnostics(
        sparse_state="absent",
        dense_state=dense_state,
        index_state="ready",
    )

    assert diagnostics.overall_state == "unavailable"


def test_retrieve_validates_limit_and_empty_query(tmp_path: Path) -> None:
    _make_vault(tmp_path)
    source = _source(
        tmp_path,
        tmp_path / "index",
        model=None,
        config=HybridConfig(dense_enabled=False),
    )

    assert source.retrieve("  ", limit=5).hits == ()
    with pytest.raises(ValueError, match="positive"):
        source.retrieve("bayes", limit=0)
