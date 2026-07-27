from __future__ import annotations

import stat
from pathlib import Path

import numpy as np

from soca.knowledge.indexing.coordinator import IndexCoordinator
from soca.knowledge.indexing.identity import CorpusSpec, EmbeddingFingerprint
from soca.knowledge.indexing.status import DenseState
from soca.knowledge.indexing.vector import stable_exact_top_k
from soca.knowledge.markdown_vault import MarkdownVaultKnowledgeSource


class FakeModel:
    model_id = "fake:v2"
    embedding_fingerprint = EmbeddingFingerprint(
        adapter="fake",
        adapter_version="test-v1",
        model_id="v2",
        dimension=2,
        passage_prefix="passage: ",
    )

    def __init__(self) -> None:
        self.document_calls: list[tuple[str, ...]] = []

    def embed_documents(self, texts: tuple[str, ...]) -> np.ndarray:
        self.document_calls.append(texts)
        return np.asarray([[float(index + 1), 1.0] for index, _ in enumerate(texts)], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        del text
        return np.asarray([1.0, 0.0], dtype=np.float32)


def _coordinator(root: Path, model: FakeModel | None = None) -> IndexCoordinator:
    reader = MarkdownVaultKnowledgeSource(root, include_globs=("wiki/**/*.md",))
    return IndexCoordinator(
        reader,
        spec=CorpusSpec(root),
        index_home=root / ".index",
        model=model,
    )


def test_snapshot_never_embeds_documents_and_dense_is_sparse_fallback(tmp_path: Path) -> None:
    (tmp_path / "wiki").mkdir()
    (tmp_path / "wiki/note.md").write_text("# Bayes\nEvidence updates beliefs.", encoding="utf-8")
    model = FakeModel()
    coordinator = _coordinator(tmp_path, model)

    snapshot = coordinator.snapshot()

    assert snapshot.sparse_index.chunks
    assert snapshot.dense_state == DenseState.ABSENT
    assert snapshot.dense_index is None
    assert model.document_calls == []


def test_dense_build_is_explicit_and_reuses_vector_after_rename(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    original = wiki / "old.md"
    original.write_text("# Bayes\nEvidence updates beliefs.", encoding="utf-8")
    model = FakeModel()
    coordinator = _coordinator(tmp_path, model)

    first = coordinator.build_blocking(dense=True)
    assert first.dense is not None
    assert first.dense.embedded_rows == 1
    assert len(model.document_calls) == 1

    original.rename(wiki / "new.md")
    coordinator.snapshot()
    assert coordinator.status().dense_state == DenseState.STALE

    second = coordinator.build_blocking(dense=True)

    assert second.dense is not None
    assert second.dense.reused_rows == 1
    assert second.dense.embedded_rows == 0
    assert len(model.document_calls) == 1


def test_edit_only_embeds_new_input_and_failed_generation_keeps_sparse_usable(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    note = wiki / "note.md"
    note.write_text("# Bayes\nEvidence updates beliefs.", encoding="utf-8")
    model = FakeModel()
    coordinator = _coordinator(tmp_path, model)
    coordinator.build_blocking(dense=True)

    note.write_text("# Bayes\nEvidence now changes beliefs.", encoding="utf-8")
    stale = coordinator.snapshot()
    assert stale.dense_state == DenseState.STALE
    assert stale.dense_index is None
    coordinator.build_blocking(dense=True)

    assert model.document_calls[-1] == ("# Bayes\nEvidence now changes beliefs.",)
    assert coordinator.snapshot().dense_state == DenseState.READY


def test_stable_exact_search_resolves_kth_boundary_by_id() -> None:
    scores = np.asarray([0.8, 0.8, 0.7, 0.8], dtype=np.float32)

    order = stable_exact_top_k(scores, ("z", "a", "b", "m"), limit=2)

    assert order == (1, 3)


def test_catalog_verify_reports_clean_generation(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "note.md").write_text("# Note\nContent.", encoding="utf-8")
    coordinator = _coordinator(tmp_path, FakeModel())
    coordinator.build_blocking(dense=True)

    assert coordinator.verify() == ()


def test_catalog_file_is_private_and_repairs_existing_mode(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "note.md").write_text("# Note\nContent.", encoding="utf-8")
    coordinator = _coordinator(tmp_path)

    coordinator.snapshot()
    catalog_path = tmp_path / ".index" / "v2" / "index.sqlite3"
    catalog_path.chmod(0o644)

    coordinator.snapshot()

    assert stat.S_IMODE(catalog_path.stat().st_mode) == 0o600
