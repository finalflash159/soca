from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
from pathlib import Path
from uuid import uuid4

import numpy as np

from soca.knowledge.index.models import MarkdownChunk
from soca.knowledge.index.persistence import ensure_private_directory, fsync_directory
from soca.knowledge.retrievers.dense import DenseIndex, EmbeddingModel

LOGGER = logging.getLogger(__name__)
DENSE_VERSION = 1
MAX_DENSE_METADATA_BYTES = 4 * 1024 * 1024
MAX_DENSE_CHUNKS = 250_000
MAX_DENSE_DIMENSION = 8_192
MAX_VECTOR_BYTES = 256 * 1024 * 1024
MAX_NPY_OVERHEAD = 1024 * 1024


def _validated_raw_vector_bytes(chunk_count: int, dimension: int) -> int:
    if not 1 <= chunk_count <= MAX_DENSE_CHUNKS:
        raise ValueError("dense chunk count is outside the safe limit")
    if not 1 <= dimension <= MAX_DENSE_DIMENSION:
        raise ValueError("dense dimension is outside the safe limit")
    size = chunk_count * dimension * np.dtype(np.float32).itemsize
    if size > MAX_VECTOR_BYTES:
        raise ValueError("dense vector matrix exceeds the safe byte limit")
    return size


def _regular_file_size(path: Path, *, max_bytes: int) -> int | None:
    try:
        metadata = path.lstat()
    except OSError:
        return None
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        return None
    if metadata.st_size > max_bytes:
        return None
    return metadata.st_size


def _generation_name(model_id: str, source_digest: str) -> str:
    model_hash = hashlib.sha256(model_id.encode("utf-8")).hexdigest()[:12]
    return f"vectors-{model_hash}-{source_digest[:16]}.npy"


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    ensure_private_directory(path.parent)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            os.chmod(path, 0o600)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def save_dense_index(directory: Path, index: DenseIndex) -> None:
    ensure_private_directory(directory)
    _validated_raw_vector_bytes(len(index.chunk_ids), index.dimension)
    generation = _generation_name(index.model_id, index.source_digest)
    vectors_path = directory / generation
    temporary = directory / f".{generation}.{uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, index.vectors, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, vectors_path)
        if os.name == "posix":
            os.chmod(vectors_path, 0o600)
        # Vector generation must be durable before dense.json can point to it.
        fsync_directory(directory)
    finally:
        temporary.unlink(missing_ok=True)

    _write_json_atomic(
        directory / "dense.json",
        {
            "dense_version": DENSE_VERSION,
            "model_id": index.model_id,
            "source_digest": index.source_digest,
            "chunk_ids": list(index.chunk_ids),
            "dimension": index.dimension,
            "vectors_file": generation,
        },
    )


def load_dense_index(
    directory: Path,
    *,
    model_id: str,
    source_digest: str | None = None,
) -> DenseIndex | None:
    metadata_path = directory / "dense.json"
    try:
        if (
            _regular_file_size(
                metadata_path,
                max_bytes=MAX_DENSE_METADATA_BYTES,
            )
            is None
        ):
            return None
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        if payload.get("dense_version") != DENSE_VERSION:
            return None
        if payload.get("model_id") != model_id:
            return None
        cached_source_digest = payload.get("source_digest")
        if not isinstance(cached_source_digest, str) or not cached_source_digest:
            return None
        if source_digest is not None and cached_source_digest != source_digest:
            return None

        chunk_ids = payload.get("chunk_ids")
        vectors_file = payload.get("vectors_file")
        dimension = payload.get("dimension")
        if (
            not isinstance(chunk_ids, list)
            or any(not isinstance(item, str) for item in chunk_ids)
            or not isinstance(vectors_file, str)
            or Path(vectors_file).name != vectors_file
            or isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or len(chunk_ids) != len(set(chunk_ids))
        ):
            return None

        raw_bytes = _validated_raw_vector_bytes(len(chunk_ids), dimension)
        vectors_path = directory / vectors_file
        file_size = _regular_file_size(
            vectors_path,
            max_bytes=MAX_VECTOR_BYTES + MAX_NPY_OVERHEAD,
        )
        if file_size is None or file_size < raw_bytes or file_size > raw_bytes + MAX_NPY_OVERHEAD:
            return None

        vectors = np.load(
            vectors_path,
            allow_pickle=False,
            mmap_mode="r",
        )
        if vectors.dtype != np.float32:
            return None
        if vectors.shape != (len(chunk_ids), dimension):
            return None
        return DenseIndex(
            model_id=model_id,
            source_digest=cached_source_digest,
            chunk_ids=tuple(chunk_ids),
            vectors=vectors,
        )
    except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
        LOGGER.warning("Ignoring invalid local dense index cache", exc_info=True)
        return None


class DenseIndexStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def load_exact(
        self,
        *,
        model_id: str,
        source_digest: str,
    ) -> DenseIndex | None:
        return load_dense_index(
            self.directory,
            model_id=model_id,
            source_digest=source_digest,
        )

    def load_latest_compatible(self, *, model_id: str) -> DenseIndex | None:
        return load_dense_index(
            self.directory,
            model_id=model_id,
        )

    def persist(self, index: DenseIndex) -> None:
        save_dense_index(self.directory, index)

    def refresh(
        self,
        chunks: tuple[MarkdownChunk, ...],
        *,
        source_digest: str,
        model: EmbeddingModel,
        previous: DenseIndex | None = None,
    ) -> DenseIndex:
        cached = previous
        if cached is None:
            cached = self.load_latest_compatible(
                model_id=model.model_id,
            )
        if (
            cached is not None
            and cached.model_id == model.model_id
            and cached.source_digest == source_digest
            and cached.chunk_ids == tuple(chunk.chunk_id for chunk in chunks)
        ):
            return cached

        reusable: dict[str, np.ndarray] = {}
        if cached is not None and cached.model_id == model.model_id:
            reusable = {
                chunk_id: cached.vectors[index] for index, chunk_id in enumerate(cached.chunk_ids)
            }

        missing = tuple(chunk for chunk in chunks if chunk.chunk_id not in reusable)
        embedded = (
            model.embed_documents(tuple(chunk.text for chunk in missing))
            if missing
            else np.empty((0, cached.dimension if cached is not None else 0), dtype=np.float32)
        )
        new_vectors = {
            **reusable,
            **{chunk.chunk_id: embedded[index] for index, chunk in enumerate(missing)},
        }
        chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
        if chunk_ids:
            matrix = np.stack([new_vectors[chunk_id] for chunk_id in chunk_ids])
        else:
            raise ValueError("cannot build a dense index without chunks")

        index = DenseIndex(
            model_id=model.model_id,
            source_digest=source_digest,
            chunk_ids=chunk_ids,
            vectors=matrix,
        )
        try:
            self.persist(index)
        except OSError:
            LOGGER.warning(
                "Dense index persistence failed; continuing with in-memory vectors",
                exc_info=True,
            )
        return index
