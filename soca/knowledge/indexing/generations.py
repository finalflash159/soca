from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np

from soca.knowledge.index.dense_persistence import MAX_NPY_OVERHEAD, MAX_VECTOR_BYTES
from soca.knowledge.index.models import VaultIndex
from soca.knowledge.index.persistence import ensure_private_directory, fsync_directory
from soca.knowledge.indexing.catalog import IndexCatalog
from soca.knowledge.indexing.identity import (
    CorpusSpec,
    embedding_fingerprint_for,
    embedding_input_hash,
)
from soca.knowledge.retrievers.dense import DenseIndex, EmbeddingModel


@dataclass(frozen=True)
class DenseBuildReport:
    generation_id: str
    source_revision: int
    source_digest: str
    row_count: int
    dimension: int
    reused_rows: int
    embedded_rows: int
    vector_file: Path
    vector_sha256: str


class DenseBuildInProgress(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_vector_file(path: Path) -> np.ndarray:
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("dense vector file must be a regular non-symlink file")
    if metadata.st_size > MAX_VECTOR_BYTES + MAX_NPY_OVERHEAD:
        raise ValueError("dense vector file exceeds the safe byte limit")
    vectors = np.load(path, allow_pickle=False, mmap_mode="r")
    if vectors.ndim != 2 or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError("dense vector file has an invalid matrix")
    return vectors


class DenseGenerationBuilder:
    def __init__(self, catalog: IndexCatalog) -> None:
        self.catalog = catalog

    def load_ready(
        self,
        spec: CorpusSpec,
        *,
        index: VaultIndex,
        revision: int,
        model: EmbeddingModel,
    ) -> DenseIndex | None:
        fingerprint = embedding_fingerprint_for(model)
        row = self.catalog.ready_generation(
            spec,
            revision=revision,
            source_digest=index.content_digest,
            embedding=fingerprint,
        )
        if row is None:
            return None
        path = self.catalog.generation_root(spec.corpus_identity) / row["vector_file"]
        try:
            vectors = _load_vector_file(path)
            rows = self.catalog.generation_rows(row["id"])
            chunk_ids = tuple(item[1] for item in rows)
            if vectors.shape != (len(chunk_ids), row["dimension"]):
                return None
            return DenseIndex(
                model_id=model.model_id,
                source_digest=index.content_digest,
                chunk_ids=chunk_ids,
                vectors=np.asarray(vectors, dtype=np.float32),
            )
        except (OSError, ValueError, EOFError):
            return None

    def build(
        self,
        spec: CorpusSpec,
        *,
        index: VaultIndex,
        revision: int,
        model: EmbeddingModel | None,
        lease_seconds: int = 120,
        force: bool = False,
    ) -> tuple[DenseIndex, DenseBuildReport]:
        if model is None:
            raise FileNotFoundError("embedding model is not provisioned")
        if not index.chunks:
            raise ValueError("cannot build a dense generation without chunks")
        fingerprint = embedding_fingerprint_for(model)
        existing = None if force else self.load_ready(spec, index=index, revision=revision, model=model)
        if existing is not None:
            row = self.catalog.ready_generation(
                spec,
                revision=revision,
                source_digest=index.content_digest,
                embedding=fingerprint,
            )
            assert row is not None
            path = self.catalog.generation_root(spec.corpus_identity) / row["vector_file"]
            return existing, DenseBuildReport(
                generation_id=row["id"],
                source_revision=revision,
                source_digest=index.content_digest,
                row_count=len(index.chunks),
                dimension=existing.dimension,
                reused_rows=row["reused_rows"],
                embedded_rows=row["embedded_rows"],
                vector_file=path,
                vector_sha256=row["vector_sha256"] or _sha256_file(path),
            )

        job = self.catalog.claim_dense_job(
            spec,
            revision=revision,
            embedding=fingerprint,
            total=len(index.chunks),
            lease_seconds=lease_seconds,
        )
        if job is None:
            raise DenseBuildInProgress("another dense builder owns this revision")
        generation_id = uuid4().hex
        generation_dir = self.catalog.generation_root(spec.corpus_identity)
        ensure_private_directory(generation_dir)
        vector_file = f"vectors-{generation_id}.npy"
        final_path = generation_dir / vector_file
        temporary = generation_dir / f".{vector_file}.{uuid4().hex}.tmp"
        try:
            reusable = self.catalog.compatible_rows(spec, fingerprint)
            old_vectors: dict[str, np.ndarray] = {}
            file_cache: dict[tuple[str, str | None], np.ndarray] = {}
            for input_hash, (source_file, row_index, source_sha256) in reusable.items():
                source_path = generation_dir / source_file
                try:
                    if source_sha256 and _sha256_file(source_path) != source_sha256:
                        continue
                    matrix = file_cache.setdefault(
                        (source_file, source_sha256),
                        _load_vector_file(source_path),
                    )
                    old_vectors[input_hash] = np.asarray(matrix[row_index], dtype=np.float32)
                except (OSError, ValueError, IndexError, EOFError):
                    continue

            hashes = tuple(embedding_input_hash(fingerprint, chunk.text) for chunk in index.chunks)
            missing_positions = tuple(
                position for position, input_hash in enumerate(hashes) if input_hash not in old_vectors
            )
            missing = tuple(index.chunks[position].text for position in missing_positions)
            encoded = (
                np.asarray(model.embed_documents(missing), dtype=np.float32)
                if missing
                else np.empty((0, 0), dtype=np.float32)
            )
            if missing:
                if encoded.ndim != 2 or encoded.shape[0] != len(missing):
                    raise ValueError("embedding backend returned an invalid matrix")
                if not np.isfinite(encoded).all():
                    raise ValueError("embedding backend returned non-finite vectors")
                norms = np.linalg.norm(encoded, axis=1, keepdims=True)
                if np.any(norms <= 1e-12):
                    raise ValueError("embedding backend returned zero-norm vectors")
                encoded = np.ascontiguousarray(encoded / norms, dtype=np.float32)
            dimension = int(encoded.shape[1]) if missing else int(next(iter(old_vectors.values())).shape[0])
            vectors = np.empty((len(index.chunks), dimension), dtype=np.float32)
            encoded_position = 0
            for position, input_hash in enumerate(hashes):
                if input_hash in old_vectors:
                    vector = old_vectors[input_hash]
                else:
                    vector = encoded[encoded_position]
                    encoded_position += 1
                if vector.shape != (dimension,):
                    raise ValueError("reused vector dimension does not match current generation")
                vectors[position] = vector
            dense = DenseIndex(
                model_id=model.model_id,
                source_digest=index.content_digest,
                chunk_ids=tuple(chunk.chunk_id for chunk in index.chunks),
                vectors=vectors,
            )
            if dense.vectors.nbytes > MAX_VECTOR_BYTES:
                raise ValueError("dense vector matrix exceeds the safe byte limit")
            self.catalog.begin_generation(
                spec,
                generation_id=generation_id,
                revision=revision,
                source_digest=index.content_digest,
                embedding=fingerprint,
                vector_file=vector_file,
                row_count=len(index.chunks),
                dimension=dimension,
                reused_rows=len(index.chunks) - len(missing_positions),
                embedded_rows=len(missing_positions),
            )
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    np.save(handle, dense.vectors, allow_pickle=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, final_path)
                os.chmod(final_path, 0o600)
                fsync_directory(generation_dir)
            finally:
                temporary.unlink(missing_ok=True)
            vector_sha256 = _sha256_file(final_path)
            rows = tuple((position, chunk.chunk_id, hashes[position]) for position, chunk in enumerate(index.chunks))
            self.catalog.publish_generation(
                spec,
                generation_id=generation_id,
                rows=rows,
                vector_sha256=vector_sha256,
                vector_bytes=final_path.stat().st_size,
                job=job,
            )
            return dense, DenseBuildReport(
                generation_id=generation_id,
                source_revision=revision,
                source_digest=index.content_digest,
                row_count=len(index.chunks),
                dimension=dimension,
                reused_rows=len(index.chunks) - len(missing_positions),
                embedded_rows=len(missing_positions),
                vector_file=final_path,
                vector_sha256=vector_sha256,
            )
        except Exception as exc:
            self.catalog.fail_generation(generation_id, "encode_failed", job)
            temporary.unlink(missing_ok=True)
            final_path.unlink(missing_ok=True)
            raise exc
