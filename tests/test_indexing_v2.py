from __future__ import annotations

import sqlite3
import stat
from pathlib import Path

import numpy as np
import pytest

from soca.knowledge.indexing import generations
from soca.knowledge.indexing.coordinator import IndexCoordinator
from soca.knowledge.indexing.generations import DenseGenerationCorrupt
from soca.knowledge.indexing.identity import (
    ChunkerFingerprint,
    CorpusSpec,
    EmbeddingFingerprint,
)
from soca.knowledge.indexing.status import DenseState
from soca.knowledge.indexing.vector import stable_exact_top_k
from soca.knowledge.indexing.watcher import IndexWatcher
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


class PreviousModel(FakeModel):
    model_id = "fake:v1"
    embedding_fingerprint = EmbeddingFingerprint(
        adapter="fake",
        adapter_version="test-v1",
        model_id="v1",
        dimension=2,
        passage_prefix="passage: ",
    )


def _coordinator(
    root: Path,
    model: FakeModel | None = None,
    *,
    chunker: ChunkerFingerprint | None = None,
) -> IndexCoordinator:
    reader = MarkdownVaultKnowledgeSource(root, include_globs=("wiki/**/*.md",))
    return IndexCoordinator(
        reader,
        spec=CorpusSpec(root),
        index_home=root / ".index",
        model=model,
        chunker=chunker,
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


def test_publishing_new_embedding_profile_retires_old_active_pointer(
    tmp_path: Path,
) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "note.md").write_text("# Note\nContent.", encoding="utf-8")
    _coordinator(tmp_path, PreviousModel()).build_blocking(dense=True)
    current = _coordinator(tmp_path, FakeModel())

    current.build_blocking(dense=True)

    assert current.verify() == ()
    pointers = current.inspect()["pointers"]
    assert isinstance(pointers, list)
    assert [pointer["embedding_fingerprint"] for pointer in pointers] == [
        FakeModel.embedding_fingerprint.value
    ]


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


def test_force_build_swaps_active_pointer_and_operator_can_rollback(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "note.md").write_text("# Note\nContent.", encoding="utf-8")
    coordinator = _coordinator(tmp_path, FakeModel())

    first = coordinator.build_blocking(dense=True)
    second = coordinator.build_blocking(dense=True, force_dense=True)

    assert first.dense is not None
    assert second.dense is not None
    assert coordinator.status().dense_generation == second.dense.generation_id
    rolled_back = coordinator.rollback()
    assert rolled_back == first.dense.generation_id
    assert coordinator.status().dense_generation == first.dense.generation_id
    assert coordinator.snapshot().dense_state == DenseState.READY


def test_corrupt_active_generation_raises_instead_of_falling_back(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "note.md").write_text("# Note\nContent.", encoding="utf-8")
    coordinator = _coordinator(tmp_path, FakeModel())
    report = coordinator.build_blocking(dense=True)
    assert report.dense is not None
    report.dense.vector_file.write_bytes(b"corrupt")

    with pytest.raises(DenseGenerationCorrupt, match="checksum"):
        coordinator.snapshot()


def test_ready_generation_reuses_checksum_until_vector_file_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "note.md").write_text("# Note\nContent.", encoding="utf-8")
    coordinator = _coordinator(tmp_path, FakeModel())
    report = coordinator.build_blocking(dense=True)
    assert report.dense is not None
    real_sha256 = generations._sha256_file
    hashed: list[Path] = []

    def tracked_sha256(path: Path) -> str:
        hashed.append(path)
        return real_sha256(path)

    monkeypatch.setattr(generations, "_sha256_file", tracked_sha256)

    coordinator.snapshot()
    coordinator.snapshot()
    report.dense.vector_file.touch()
    coordinator.snapshot()

    assert hashed == [report.dense.vector_file, report.dense.vector_file]


def test_dense_builder_batches_embedding_and_watcher_rebuilds_changes(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    for index in range(33):
        (wiki / f"{index:02d}.md").write_text(
            f"# Note {index}\nContent {index}.",
            encoding="utf-8",
        )
    model = FakeModel()
    coordinator = _coordinator(tmp_path, model)
    watcher = IndexWatcher(coordinator)

    status = watcher.reconcile()

    assert status.dense_state == DenseState.READY
    assert [len(batch) for batch in model.document_calls] == [32, 1]
    assert watcher.last_error is None


def test_chunker_fingerprint_change_rechunks_unchanged_documents(
    tmp_path: Path,
) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "note.md").write_text(
        "# Note\n" + "\n".join(f"line {index} alpha beta gamma" for index in range(40)),
        encoding="utf-8",
    )
    initial = _coordinator(
        tmp_path,
        chunker=ChunkerFingerprint(target_tokens=320),
    ).snapshot()
    changed = _coordinator(
        tmp_path,
        chunker=ChunkerFingerprint(
            algorithm_version="chunker-v2",
            target_tokens=32,
        ),
    ).snapshot()

    assert len(initial.sparse_index.chunks) == 1
    assert len(changed.sparse_index.chunks) > 1
    assert changed.revision == initial.revision + 1


def test_equivalent_rechunk_still_persists_new_chunker_fingerprint(
    tmp_path: Path,
) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "note.md").write_text("# Note\nshort content", encoding="utf-8")
    _coordinator(
        tmp_path,
        chunker=ChunkerFingerprint(algorithm_version="old"),
    ).snapshot()
    current = ChunkerFingerprint(algorithm_version="new")

    _coordinator(tmp_path, chunker=current).snapshot()

    database = tmp_path / ".index" / "v2" / "index.sqlite3"
    connection = sqlite3.connect(database)
    stored = connection.execute(
        "SELECT chunker_fingerprint FROM corpora"
    ).fetchone()[0]
    connection.close()
    assert stored == current.value


def test_schema_v2_upgrade_restores_active_generation_pointer(tmp_path: Path) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "note.md").write_text("# Note\nContent.", encoding="utf-8")
    coordinator = _coordinator(tmp_path, FakeModel())
    report = coordinator.build_blocking(dense=True)
    assert report.dense is not None
    database = tmp_path / ".index" / "v2" / "index.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("DROP TABLE dense_generation_pointers")
    connection.execute("PRAGMA user_version = 2")
    connection.commit()
    connection.close()

    migrated = _coordinator(tmp_path, FakeModel())

    assert migrated.status().dense_generation == report.dense.generation_id
    assert migrated.snapshot().dense_state == DenseState.READY


def test_gc_keeps_active_and_previous_generation_during_rollback_window(
    tmp_path: Path,
) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "note.md").write_text("# Note\nContent.", encoding="utf-8")
    coordinator = _coordinator(tmp_path, FakeModel())
    first = coordinator.build_blocking(dense=True)
    second = coordinator.build_blocking(dense=True, force_dense=True)
    third = coordinator.build_blocking(dense=True, force_dense=True)
    assert first.dense and second.dense and third.dense
    database = tmp_path / ".index" / "v2" / "index.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute(
        "UPDATE dense_generations SET completed_at='2000-01-01T00:00:00+00:00'"
    )
    connection.commit()
    connection.close()

    candidates = coordinator.gc()

    assert str(first.dense.vector_file) in candidates
    assert str(second.dense.vector_file) not in candidates
    assert str(third.dense.vector_file) not in candidates
