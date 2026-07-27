from __future__ import annotations

import json
import logging
import os
import stat
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from soca.knowledge.index import dense_persistence
from soca.knowledge.index.dense_persistence import (
    DENSE_VERSION,
    MAX_DENSE_METADATA_BYTES,
    MAX_NPY_OVERHEAD,
    MAX_VECTOR_BYTES,
    DenseIndexStore,
    load_dense_index,
    save_dense_index,
)
from soca.knowledge.index.models import MarkdownChunk
from soca.knowledge.retrievers.dense import DenseIndex


class FakeEmbeddingModel:
    def __init__(self, *, model_id: str = "fake:model") -> None:
        self._model_id = model_id
        self.document_calls: list[tuple[str, ...]] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    def embed_documents(self, texts: tuple[str, ...]) -> np.ndarray:
        self.document_calls.append(texts)
        return np.array(
            [[float(sum(text.encode("utf-8")) % 97 + 1), 1.0] for text in texts],
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> np.ndarray:
        raise AssertionError(f"index refresh must not embed a query: {text!r}")


def _index(
    *,
    source_digest: str = "a" * 64,
    model_id: str = "fake:model",
    chunk_ids: tuple[str, ...] = ("chunk-a", "chunk-b"),
) -> DenseIndex:
    return DenseIndex(
        model_id=model_id,
        source_digest=source_digest,
        chunk_ids=chunk_ids,
        vectors=np.array(
            [[3.0, 4.0], [0.0, 2.0]][: len(chunk_ids)],
            dtype=np.float32,
        ),
    )


def _chunk(chunk_id: str, text: str) -> MarkdownChunk:
    return MarkdownChunk(
        chunk_id=chunk_id,
        document_path="wiki/note.md",
        title="Note",
        tags=("test",),
        text=text,
        line_start=1,
        line_end=1,
    )


def _metadata(directory: Path) -> dict[str, Any]:
    payload = json.loads((directory / "dense.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_metadata(directory: Path, payload: object) -> None:
    (directory / "dense.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_dense_index_round_trip_uses_safe_numpy_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_calls: list[bool] = []
    load_calls: list[tuple[bool, str | None]] = []
    original_save = dense_persistence.np.save
    original_load = dense_persistence.np.load

    def record_save(
        file: object,
        array: np.ndarray,
        *,
        allow_pickle: bool,
    ) -> None:
        save_calls.append(allow_pickle)
        original_save(file, array, allow_pickle=allow_pickle)

    def record_load(
        file: object,
        *,
        allow_pickle: bool,
        mmap_mode: str | None,
    ) -> np.ndarray:
        load_calls.append((allow_pickle, mmap_mode))
        return original_load(
            file,
            allow_pickle=allow_pickle,
            mmap_mode=mmap_mode,
        )

    monkeypatch.setattr(dense_persistence.np, "save", record_save)
    monkeypatch.setattr(dense_persistence.np, "load", record_load)
    expected = _index()

    save_dense_index(tmp_path, expected)
    loaded = load_dense_index(
        tmp_path,
        model_id=expected.model_id,
        source_digest=expected.source_digest,
    )

    assert loaded is not None
    assert loaded.model_id == expected.model_id
    assert loaded.source_digest == expected.source_digest
    assert loaded.chunk_ids == expected.chunk_ids
    np.testing.assert_allclose(loaded.vectors, expected.vectors)
    assert not loaded.vectors.flags.writeable
    assert save_calls == [False]
    assert load_calls == [(False, "r")]


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits are unavailable")
def test_dense_cache_uses_private_directory_and_file_modes(tmp_path: Path) -> None:
    save_dense_index(tmp_path, _index())
    payload = _metadata(tmp_path)
    vectors_path = tmp_path / payload["vectors_file"]

    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
    assert stat.S_IMODE((tmp_path / "dense.json").stat().st_mode) == 0o600
    assert stat.S_IMODE(vectors_path.stat().st_mode) == 0o600


def test_missing_dense_metadata_is_a_silent_cache_miss(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(
        logging.WARNING,
        logger="soca.knowledge.index.dense_persistence",
    ):
        loaded = load_dense_index(tmp_path, model_id="fake:model")

    assert loaded is None
    assert caplog.records == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dense_version", DENSE_VERSION + 1),
        ("model_id", "different:model"),
        ("source_digest", ""),
        ("chunk_ids", ["chunk-a", "chunk-a"]),
        ("dimension", True),
        ("dimension", 0),
        ("vectors_file", "../outside.npy"),
    ],
)
def test_invalid_dense_metadata_is_a_cache_miss(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    index = _index()
    save_dense_index(tmp_path, index)
    payload = _metadata(tmp_path)
    payload[field] = value
    _write_metadata(tmp_path, payload)

    assert load_dense_index(tmp_path, model_id=index.model_id) is None


def test_wrong_expected_source_digest_is_a_cache_miss(tmp_path: Path) -> None:
    index = _index()
    save_dense_index(tmp_path, index)

    assert (
        load_dense_index(
            tmp_path,
            model_id=index.model_id,
            source_digest="b" * 64,
        )
        is None
    )


def test_corrupt_json_and_object_array_are_cache_misses(tmp_path: Path) -> None:
    index = _index(chunk_ids=("chunk-a",))
    save_dense_index(tmp_path, index)
    metadata_path = tmp_path / "dense.json"
    metadata_path.write_text("{invalid", encoding="utf-8")
    assert load_dense_index(tmp_path, model_id=index.model_id) is None

    save_dense_index(tmp_path, index)
    payload = _metadata(tmp_path)
    vectors_path = tmp_path / payload["vectors_file"]
    np.save(vectors_path, np.array([["unsafe"]], dtype=object), allow_pickle=True)
    assert load_dense_index(tmp_path, model_id=index.model_id) is None


def test_dense_loader_rejects_symlink_metadata(tmp_path: Path) -> None:
    index = _index()
    save_dense_index(tmp_path, index)
    metadata_path = tmp_path / "dense.json"
    target = tmp_path / "metadata-target.json"
    target.write_bytes(metadata_path.read_bytes())
    metadata_path.unlink()
    metadata_path.symlink_to(target)

    assert load_dense_index(tmp_path, model_id=index.model_id) is None


@pytest.mark.parametrize("replacement", ["symlink", "directory"])
def test_dense_loader_rejects_non_regular_vector_file(
    tmp_path: Path,
    replacement: str,
) -> None:
    index = _index()
    save_dense_index(tmp_path, index)
    payload = _metadata(tmp_path)
    vectors_path = tmp_path / payload["vectors_file"]
    original = vectors_path.read_bytes()
    vectors_path.unlink()
    if replacement == "symlink":
        target = tmp_path / "vectors-target.npy"
        target.write_bytes(original)
        vectors_path.symlink_to(target)
    else:
        vectors_path.mkdir()

    assert load_dense_index(tmp_path, model_id=index.model_id) is None


def test_oversized_metadata_is_rejected_before_numpy_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_path = tmp_path / "dense.json"
    metadata_path.write_bytes(b"x" * (MAX_DENSE_METADATA_BYTES + 1))

    def fail_load(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("np.load must not run for oversized metadata")

    monkeypatch.setattr(dense_persistence.np, "load", fail_load)

    assert load_dense_index(tmp_path, model_id="fake:model") is None


def test_oversized_vector_file_is_rejected_before_numpy_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _index()
    save_dense_index(tmp_path, index)
    payload = _metadata(tmp_path)
    vectors_path = tmp_path / payload["vectors_file"]
    with vectors_path.open("wb") as handle:
        handle.truncate(MAX_VECTOR_BYTES + MAX_NPY_OVERHEAD + 1)

    def fail_load(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("np.load must not run for oversized vectors")

    monkeypatch.setattr(dense_persistence.np, "load", fail_load)

    assert load_dense_index(tmp_path, model_id=index.model_id) is None


def test_restart_then_one_changed_chunk_only_embeds_the_changed_chunk(
    tmp_path: Path,
) -> None:
    initial_model = FakeEmbeddingModel()
    initial_chunks = (
        _chunk("chunk-a", "alpha"),
        _chunk("chunk-b", "beta"),
    )
    first = DenseIndexStore(tmp_path).refresh(
        initial_chunks,
        source_digest="a" * 64,
        model=initial_model,
    )
    restarted_model = FakeEmbeddingModel()
    changed_chunks = (
        _chunk("chunk-a", "alpha"),
        _chunk("chunk-c", "gamma changed"),
    )

    second = DenseIndexStore(tmp_path).refresh(
        changed_chunks,
        source_digest="b" * 64,
        model=restarted_model,
    )

    assert initial_model.document_calls == [("alpha", "beta")]
    assert restarted_model.document_calls == [("gamma changed",)]
    assert second.chunk_ids == ("chunk-a", "chunk-c")
    np.testing.assert_allclose(second.vectors[0], first.vectors[0])
    assert (
        DenseIndexStore(tmp_path).load_exact(
            model_id=second.model_id,
            source_digest=second.source_digest,
        )
        is not None
    )


def test_exact_cached_index_is_reused_without_embedding_or_persisting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = (_chunk("chunk-a", "alpha"),)
    original_model = FakeEmbeddingModel()
    store = DenseIndexStore(tmp_path)
    original = store.refresh(
        chunks,
        source_digest="a" * 64,
        model=original_model,
    )
    restarted_model = FakeEmbeddingModel()
    restarted_store = DenseIndexStore(tmp_path)

    def fail_persist(index: DenseIndex) -> None:
        raise AssertionError(f"exact cache hit must not be persisted: {index!r}")

    monkeypatch.setattr(restarted_store, "persist", fail_persist)

    loaded = restarted_store.refresh(
        chunks,
        source_digest="a" * 64,
        model=restarted_model,
    )

    assert loaded.chunk_ids == original.chunk_ids
    np.testing.assert_allclose(loaded.vectors, original.vectors)
    assert restarted_model.document_calls == []


def test_model_change_reembeds_every_chunk(tmp_path: Path) -> None:
    chunks = (
        _chunk("chunk-a", "alpha"),
        _chunk("chunk-b", "beta"),
    )
    DenseIndexStore(tmp_path).refresh(
        chunks,
        source_digest="a" * 64,
        model=FakeEmbeddingModel(model_id="fake:old"),
    )
    new_model = FakeEmbeddingModel(model_id="fake:new")

    refreshed = DenseIndexStore(tmp_path).refresh(
        chunks,
        source_digest="a" * 64,
        model=new_model,
    )

    assert refreshed.model_id == "fake:new"
    assert new_model.document_calls == [("alpha", "beta")]


def test_persist_failure_keeps_the_new_dense_index_in_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = DenseIndexStore(tmp_path)

    def fail_persist(index: DenseIndex) -> None:
        raise OSError(f"disk unavailable for {index.source_digest}")

    monkeypatch.setattr(store, "persist", fail_persist)
    model = FakeEmbeddingModel()

    index = store.refresh(
        (_chunk("chunk-a", "alpha"),),
        source_digest="a" * 64,
        model=model,
    )

    assert index.chunk_ids == ("chunk-a",)
    assert model.document_calls == [("alpha",)]


def test_refresh_rejects_an_empty_chunk_collection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="without chunks"):
        DenseIndexStore(tmp_path).refresh(
            (),
            source_digest="a" * 64,
            model=FakeEmbeddingModel(),
        )
