import time
from pathlib import Path

import numpy as np

from soca.knowledge.indexing.coordinator import IndexCoordinator
from soca.knowledge.indexing.identity import CorpusSpec, EmbeddingFingerprint
from soca.knowledge.indexing.watcher import IndexWatcher
from soca.knowledge.markdown_vault import MarkdownVaultKnowledgeSource


class _WatcherEmbedding:
    model_id = "test:watcher"
    embedding_fingerprint = EmbeddingFingerprint(
        adapter="test",
        adapter_version="1",
        model_id="watcher",
        dimension=4,
    )

    def embed_documents(self, texts: tuple[str, ...]) -> np.ndarray:
        return np.asarray(
            [np.resize(np.frombuffer(text.encode(), dtype=np.uint8), 4) for text in texts],
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> np.ndarray:
        del text
        return np.ones(4, dtype=np.float32)


def _wait_for_revision(coordinator: IndexCoordinator, revision: int) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if coordinator.status().revision >= revision:
            return
        time.sleep(0.05)
    raise AssertionError(f"watcher did not reach revision {revision}")


def test_watcher_reconciles_add_edit_delete_and_stops(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    wiki = vault / "wiki"
    wiki.mkdir(parents=True)
    note = wiki / "note.md"
    note.write_text("# Note\nfirst", encoding="utf-8")
    reader = MarkdownVaultKnowledgeSource(vault, include_globs=("wiki/**/*.md",))
    coordinator = IndexCoordinator(
        reader,
        spec=CorpusSpec(vault),
        index_home=tmp_path / "index",
        model=_WatcherEmbedding(),
    )
    coordinator.build_blocking(dense=True)
    initial_revision = coordinator.status().revision
    watcher = IndexWatcher(coordinator, interval_seconds=0.05)
    watcher.start()
    try:
        note.write_text("# Note\nsecond", encoding="utf-8")
        _wait_for_revision(coordinator, initial_revision + 1)
        edited_revision = coordinator.status().revision
        note.unlink()
        _wait_for_revision(coordinator, edited_revision + 1)
        added = wiki / "added.md"
        added.write_text("# Added\nthird", encoding="utf-8")
        _wait_for_revision(coordinator, edited_revision + 2)
        assert watcher.last_error is None
    finally:
        watcher.stop()
    assert watcher._thread is None

