from __future__ import annotations

import os
from pathlib import Path

import pytest

from soca.knowledge.base import KnowledgeDocument
from soca.knowledge.index.models import VaultIndex
from soca.knowledge.index.vault_index import VaultIndexer, VaultIndexStore
from soca.knowledge.markdown_vault import MarkdownVaultKnowledgeSource


class CountingVaultSource(MarkdownVaultKnowledgeSource):
    def __init__(self, root: Path) -> None:
        super().__init__(root, include_globs=("wiki/**/*.md",))
        self.read_paths: list[str] = []

    def read(self, path: str) -> KnowledgeDocument:
        self.read_paths.append(path)
        return super().read(path)


def _make_vault(tmp_path: Path) -> Path:
    vault = tmp_path / "vault"
    wiki = vault / "wiki"
    wiki.mkdir(parents=True)
    (wiki / "a.md").write_text("# Alpha\nalpha body", encoding="utf-8")
    (wiki / "b.md").write_text("# Beta\nbeta body", encoding="utf-8")
    return vault


def test_first_build_reads_and_indexes_every_markdown_file(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    reader = CountingVaultSource(vault)
    store = VaultIndexStore(index_home=tmp_path / "index-home")

    index = VaultIndexer(reader, store).refresh()

    assert reader.read_paths == ["wiki/a.md", "wiki/b.md"]
    assert [document.path for document in index.documents] == [
        "wiki/a.md",
        "wiki/b.md",
    ]
    assert index.chunks
    assert store.manifest_path_for(vault).is_file()


def test_new_store_and_source_reuse_persisted_documents_without_reads(
    tmp_path: Path,
) -> None:
    vault = _make_vault(tmp_path)
    first_reader = CountingVaultSource(vault)
    first = VaultIndexer(
        first_reader,
        VaultIndexStore(index_home=tmp_path / "index-home"),
    ).refresh()
    assert first_reader.read_paths == ["wiki/a.md", "wiki/b.md"]

    restarted_reader = CountingVaultSource(vault)
    restarted = VaultIndexer(
        restarted_reader,
        VaultIndexStore(index_home=tmp_path / "index-home"),
    ).refresh()

    assert restarted_reader.read_paths == []
    assert restarted == first


def test_refresh_reads_only_the_changed_file(tmp_path: Path) -> None:
    vault = _make_vault(tmp_path)
    reader = CountingVaultSource(vault)
    indexer = VaultIndexer(
        reader,
        VaultIndexStore(index_home=tmp_path / "index-home"),
    )
    first = indexer.refresh()
    reader.read_paths.clear()
    (vault / "wiki" / "b.md").write_text(
        "# Beta\nbeta body changed and made longer",
        encoding="utf-8",
    )

    second = indexer.refresh(previous=first)

    assert reader.read_paths == ["wiki/b.md"]
    assert second.document_by_path("wiki/a.md") is first.document_by_path("wiki/a.md")
    assert "changed" in second.document_by_path("wiki/b.md").text  # type: ignore[union-attr]


def test_refresh_detects_same_size_replacement_with_preserved_mtime(
    tmp_path: Path,
) -> None:
    vault = _make_vault(tmp_path)
    reader = CountingVaultSource(vault)
    indexer = VaultIndexer(
        reader,
        VaultIndexStore(index_home=tmp_path / "index-home"),
    )
    first = indexer.refresh()
    original = vault / "wiki" / "a.md"
    original_stat = original.stat()
    replacement = vault / "wiki" / "replacement.md"
    replacement.write_text("# Alpha\nomega body", encoding="utf-8")
    os.utime(
        replacement,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    replacement.replace(original)
    reader.read_paths.clear()

    second = indexer.refresh(previous=first)

    assert reader.read_paths == ["wiki/a.md"]
    assert second.document_by_path("wiki/a.md") is not first.document_by_path("wiki/a.md")
    assert second.document_by_path("wiki/a.md").text == "# Alpha\nomega body"  # type: ignore[union-attr]


def test_forced_verification_detects_content_change_with_unchanged_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from soca.knowledge.index import vault_index as vault_index_module

    vault = _make_vault(tmp_path)
    reader = CountingVaultSource(vault)
    indexer = VaultIndexer(
        reader,
        VaultIndexStore(index_home=tmp_path / "index-home"),
    )
    first = indexer.refresh()
    fingerprints = {record.document.path: record.fingerprint for record in first.records}

    def unchanged_probe(
        _reader: CountingVaultSource,
        relative_path: str,
    ) -> vault_index_module._FileProbe:
        fingerprint = fingerprints[relative_path]
        return vault_index_module._FileProbe(
            path=fingerprint.path,
            mtime_ns=fingerprint.mtime_ns,
            ctime_ns=fingerprint.ctime_ns,
            size=fingerprint.size,
            inode=fingerprint.inode,
        )

    (vault / "wiki" / "a.md").write_text(
        "# Alpha\nomega body",
        encoding="utf-8",
    )
    monkeypatch.setattr(vault_index_module, "_probe_file", unchanged_probe)
    reader.read_paths.clear()

    verified = indexer.refresh(previous=first, verify_content=True)

    assert reader.read_paths == ["wiki/a.md", "wiki/b.md"]
    assert verified.document_by_path("wiki/a.md").text == "# Alpha\nomega body"  # type: ignore[union-attr]


def test_refresh_drops_deleted_file_without_rereading_unchanged_files(
    tmp_path: Path,
) -> None:
    vault = _make_vault(tmp_path)
    reader = CountingVaultSource(vault)
    indexer = VaultIndexer(
        reader,
        VaultIndexStore(index_home=tmp_path / "index-home"),
    )
    first = indexer.refresh()
    reader.read_paths.clear()
    (vault / "wiki" / "b.md").unlink()

    second = indexer.refresh(previous=first)

    assert reader.read_paths == []
    assert [document.path for document in second.documents] == ["wiki/a.md"]


def test_refresh_skips_file_deleted_during_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from soca.knowledge.index import vault_index as vault_index_module

    reader = CountingVaultSource(_make_vault(tmp_path))
    indexer = VaultIndexer(reader, VaultIndexStore(index_home=tmp_path / "index-home"))
    original_probe = vault_index_module._probe_file

    def race_probe(reader_arg, relative_path: str):
        if relative_path == "wiki/b.md":
            raise FileNotFoundError(relative_path)
        return original_probe(reader_arg, relative_path)

    monkeypatch.setattr(vault_index_module, "_probe_file", race_probe)

    index = indexer.refresh()

    assert [document.path for document in index.documents] == ["wiki/a.md"]


def test_refresh_skips_file_that_becomes_unreadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = CountingVaultSource(_make_vault(tmp_path))
    indexer = VaultIndexer(reader, VaultIndexStore(index_home=tmp_path / "index-home"))
    original_read = reader.read

    def race_read(path: str) -> KnowledgeDocument:
        if path == "wiki/b.md":
            raise PermissionError(path)
        return original_read(path)

    monkeypatch.setattr(reader, "read", race_read)

    index = indexer.refresh()

    assert [document.path for document in index.documents] == ["wiki/a.md"]


def test_persist_failure_returns_and_reuses_new_in_memory_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = CountingVaultSource(_make_vault(tmp_path))
    store = VaultIndexStore(index_home=tmp_path / "index-home")
    indexer = VaultIndexer(reader, store)

    def fail_persist(_index: VaultIndex) -> None:
        raise PermissionError("read-only cache")

    monkeypatch.setattr(store, "persist", fail_persist)
    first = indexer.refresh()
    reader.read_paths.clear()
    second = indexer.refresh(previous=first)

    assert second is first
    assert reader.read_paths == []
