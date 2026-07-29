from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np

from soca.knowledge.base import KnowledgeDocument
from soca.knowledge.index.models import FileFingerprint, IndexedFile, MarkdownChunk, VaultIndex
from soca.knowledge.indexing.identity import (
    ChunkerFingerprint,
    CorpusIdentity,
    CorpusSpec,
    EmbeddingFingerprint,
    canonical_json,
)
from soca.knowledge.indexing.scanner import ScanReport, VaultReader, scan_vault
from soca.knowledge.indexing.schema import catalog_path, connect_catalog, generation_root
from soca.knowledge.indexing.status import DenseState, IndexStatus, SparseState


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class SparseSyncResult:
    index: VaultIndex
    revision: int
    changed: bool
    added: int
    removed: int
    metadata_only: int


@dataclass(frozen=True)
class DenseJob:
    job_id: str
    corpus_id: str
    requested_revision: int
    embedding_fingerprint: str
    owner_instance_id: str


class IndexCatalog:
    """SQLite catalog with short transactions and immutable vector metadata."""

    def __init__(self, index_home: Path) -> None:
        self.index_home = index_home.expanduser()
        self.db_path = catalog_path(self.index_home)

    def _connect(self) -> sqlite3.Connection:
        return connect_catalog(self.db_path)

    def sparse_index(self, identity: CorpusIdentity) -> VaultIndex | None:
        connection = self._connect()
        try:
            corpus = connection.execute(
                "SELECT vault_path, current_revision FROM corpora WHERE id = ?",
                (identity.value,),
            ).fetchone()
            if corpus is None:
                return None
            files = connection.execute(
                "SELECT * FROM files WHERE corpus_id = ? ORDER BY path",
                (identity.value,),
            ).fetchall()
            chunks = connection.execute(
                "SELECT * FROM chunks WHERE corpus_id = ? ORDER BY document_path, ordinal",
                (identity.value,),
            ).fetchall()
            chunks_by_path: dict[str, list[MarkdownChunk]] = {}
            for row in chunks:
                chunk = MarkdownChunk(
                    chunk_id=row["chunk_id"],
                    document_path=row["document_path"],
                    title=row["title"],
                    tags=tuple(json.loads(row["tags_json"])),
                    text=row["text"],
                    line_start=row["line_start"],
                    line_end=row["line_end"],
                )
                chunks_by_path.setdefault(chunk.document_path, []).append(chunk)
            records: list[IndexedFile] = []
            for row in files:
                document = KnowledgeDocument(
                    id=row["path"],
                    path=row["path"],
                    title=row["title"],
                    text=row["text"],
                    tags=tuple(json.loads(row["tags_json"])),
                    frontmatter=json.loads(row["frontmatter_json"]),
                )
                fingerprint = FileFingerprint(
                    path=row["path"],
                    mtime_ns=row["mtime_ns"],
                    ctime_ns=row["ctime_ns"],
                    size=row["size"],
                    inode=row["inode"],
                    content_sha256=row["content_sha256"],
                )
                records.append(
                    IndexedFile(
                        fingerprint=fingerprint,
                        document=document,
                        chunks=tuple(chunks_by_path.get(row["path"], ())),
                    )
                )
            return VaultIndex(vault_path=corpus["vault_path"], records=tuple(records))
        finally:
            connection.close()

    def sync_sparse(
        self,
        spec: CorpusSpec,
        reader: VaultReader,
        *,
        chunker: ChunkerFingerprint | None = None,
        verify_content: bool = False,
    ) -> SparseSyncResult:
        identity = spec.corpus_identity
        resolved_chunker = chunker or ChunkerFingerprint()
        previous = self.sparse_index(identity)
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT chunker_fingerprint FROM corpora WHERE id=?",
                (identity.value,),
            ).fetchone()
        finally:
            connection.close()
        force_rechunk = (
            row is not None and row["chunker_fingerprint"] != resolved_chunker.value
        )
        report: ScanReport = scan_vault(
            reader,
            previous=previous,
            verify_content=verify_content,
            force_rechunk=force_rechunk,
            target_tokens=resolved_chunker.target_tokens,
            overlap_lines=resolved_chunker.overlap_lines,
        )
        changed = previous is None or previous.content_digest != report.index.content_digest
        old_revision = self._revision(identity)
        if previous is not None and report.index == previous and not force_rechunk:
            return SparseSyncResult(
                index=previous,
                revision=old_revision,
                changed=False,
                added=0,
                removed=0,
                metadata_only=0,
            )
        revision = old_revision + 1 if changed else old_revision
        now = _now()
        chunker_value = resolved_chunker.value
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO corpora(
                    id, kind, vault_path, policy_json, chunker_fingerprint,
                    current_revision, content_digest, created_at, updated_at,
                    sparse_state, sparse_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready', NULL)
                ON CONFLICT(id) DO UPDATE SET
                    vault_path=excluded.vault_path,
                    kind=excluded.kind,
                    policy_json=excluded.policy_json,
                    chunker_fingerprint=excluded.chunker_fingerprint,
                    current_revision=excluded.current_revision,
                    content_digest=excluded.content_digest,
                    updated_at=excluded.updated_at,
                    sparse_state='ready',
                    sparse_error=NULL
                """,
                (
                    identity.value,
                    identity.kind,
                    identity.vault_path,
                    identity.policy_json,
                    chunker_value,
                    revision,
                    report.index.content_digest,
                    now,
                    now,
                ),
            )
            current_paths = {record.document.path for record in report.index.records}
            if current_paths:
                placeholders = ",".join("?" for _ in current_paths)
                connection.execute(
                    f"DELETE FROM files WHERE corpus_id = ? AND path NOT IN ({placeholders})",
                    (identity.value, *sorted(current_paths)),
                )
            else:
                connection.execute("DELETE FROM files WHERE corpus_id = ?", (identity.value,))
            connection.execute("DELETE FROM chunks WHERE corpus_id = ?", (identity.value,))
            for record in report.index.records:
                document = record.document
                connection.execute(
                    """
                    INSERT INTO files(
                        corpus_id, path, mtime_ns, ctime_ns, size, inode,
                        content_sha256, title, tags_json, frontmatter_json, text, revision
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(corpus_id, path) DO UPDATE SET
                        mtime_ns=excluded.mtime_ns, ctime_ns=excluded.ctime_ns,
                        size=excluded.size, inode=excluded.inode,
                        content_sha256=excluded.content_sha256, title=excluded.title,
                        tags_json=excluded.tags_json, frontmatter_json=excluded.frontmatter_json,
                        text=excluded.text, revision=excluded.revision
                    """,
                    (
                        identity.value,
                        record.fingerprint.path,
                        record.fingerprint.mtime_ns,
                        record.fingerprint.ctime_ns,
                        record.fingerprint.size,
                        record.fingerprint.inode,
                        record.fingerprint.content_sha256,
                        document.title,
                        canonical_json(list(document.tags)),
                        canonical_json(dict(document.frontmatter)),
                        document.text,
                        revision,
                    ),
                )
                for ordinal, chunk in enumerate(record.chunks):
                    connection.execute(
                        """
                        INSERT INTO chunks(
                            corpus_id, chunk_id, document_path, ordinal,
                            line_start, line_end, title, tags_json, text,
                            content_sha256, revision
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            identity.value,
                            chunk.chunk_id,
                            chunk.document_path,
                            ordinal,
                            chunk.line_start,
                            chunk.line_end,
                            chunk.title,
                            canonical_json(list(chunk.tags)),
                            chunk.text,
                            hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
                            revision,
                        ),
                    )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

        return SparseSyncResult(
            index=report.index,
            revision=revision,
            changed=changed,
            added=report.added,
            removed=report.removed,
            metadata_only=report.metadata_only,
        )

    def heartbeat(self, job: DenseJob, *, lease_seconds: int = 120) -> bool:
        connection = self._connect()
        try:
            result = connection.execute(
                """
                UPDATE jobs
                SET state='BUILDING', heartbeat_at=?, lease_expires_at=datetime('now', ?), updated_at=?
                WHERE job_id=? AND owner_instance_id=? AND state IN ('CLAIMED', 'BUILDING')
                """,
                (_now(), f"+{lease_seconds} seconds", _now(), job.job_id, job.owner_instance_id),
            )
            return result.rowcount == 1
        finally:
            connection.close()

    def update_job_progress(
        self,
        job: DenseJob,
        *,
        completed: int,
        reused: int,
        lease_seconds: int = 120,
    ) -> None:
        connection = self._connect()
        try:
            updated = connection.execute(
                """
                UPDATE jobs
                SET state='BUILDING', completed=?, reused=?, heartbeat_at=?,
                    lease_expires_at=datetime('now', ?), updated_at=?
                WHERE job_id=? AND owner_instance_id=?
                  AND state IN ('CLAIMED', 'BUILDING')
                """,
                (
                    completed,
                    reused,
                    _now(),
                    f"+{lease_seconds} seconds",
                    _now(),
                    job.job_id,
                    job.owner_instance_id,
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError("dense build lease is no longer owned")
        finally:
            connection.close()

    def import_sparse_index(
        self,
        spec: CorpusSpec,
        index: VaultIndex,
        *,
        chunker: ChunkerFingerprint | None = None,
    ) -> bool:
        """Seed v2 from a validated v1 snapshot without touching the vault.

        The next normal sparse sync remains authoritative and can replace this
        cache hint if the source changed while migration was running.
        """
        identity = spec.corpus_identity
        if self.sparse_index(identity) is not None:
            return False
        now = _now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO corpora(
                    id, kind, vault_path, policy_json, chunker_fingerprint,
                    current_revision, content_digest, created_at, updated_at,
                    sparse_state, sparse_error
                ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, 'ready', NULL)
                """,
                (
                    identity.value,
                    identity.kind,
                    identity.vault_path,
                    identity.policy_json,
                    (chunker or ChunkerFingerprint()).value,
                    index.content_digest,
                    now,
                    now,
                ),
            )
            for record in index.records:
                document = record.document
                connection.execute(
                    "INSERT INTO files(corpus_id, path, mtime_ns, ctime_ns, size, inode, content_sha256, title, tags_json, frontmatter_json, text, revision) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
                    (
                        identity.value,
                        record.fingerprint.path,
                        record.fingerprint.mtime_ns,
                        record.fingerprint.ctime_ns,
                        record.fingerprint.size,
                        record.fingerprint.inode,
                        record.fingerprint.content_sha256,
                        document.title,
                        canonical_json(list(document.tags)),
                        canonical_json(dict(document.frontmatter)),
                        document.text,
                    ),
                )
                for ordinal, chunk in enumerate(record.chunks):
                    connection.execute(
                        "INSERT INTO chunks(corpus_id, chunk_id, document_path, ordinal, line_start, line_end, title, tags_json, text, content_sha256, revision) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
                        (
                            identity.value,
                            chunk.chunk_id,
                            chunk.document_path,
                            ordinal,
                            chunk.line_start,
                            chunk.line_end,
                            chunk.title,
                            canonical_json(list(chunk.tags)),
                            chunk.text,
                            hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
                        ),
                    )
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _revision(self, identity: CorpusIdentity) -> int:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT current_revision FROM corpora WHERE id = ?",
                (identity.value,),
            ).fetchone()
            return int(row[0]) if row is not None else 0
        finally:
            connection.close()

    def status(
        self,
        spec: CorpusSpec,
        *,
        embedding_fingerprint: EmbeddingFingerprint | None = None,
    ) -> IndexStatus:
        identity = spec.corpus_identity
        connection = self._connect()
        try:
            corpus = connection.execute("SELECT * FROM corpora WHERE id = ?", (identity.value,)).fetchone()
            if corpus is None:
                return IndexStatus(
                    corpus_id=identity.value,
                    corpus_kind=identity.kind,
                    vault_path=identity.vault_path,
                    sparse_state=SparseState.ABSENT,
                    dense_state=(DenseState.MODEL_MISSING if embedding_fingerprint is None else DenseState.ABSENT),
                    revision=0,
                    content_digest=None,
                    documents=0,
                    chunks=0,
                )
            documents = connection.execute(
                "SELECT COUNT(*) FROM files WHERE corpus_id = ?", (identity.value,)
            ).fetchone()[0]
            chunks = connection.execute(
                "SELECT COUNT(*) FROM chunks WHERE corpus_id = ?", (identity.value,)
            ).fetchone()[0]
            dense = None
            if embedding_fingerprint is not None:
                dense = connection.execute(
                    """
                    SELECT g.* FROM dense_generation_pointers p
                    JOIN dense_generations g ON g.id=p.active_generation_id
                    WHERE p.corpus_id = ? AND p.embedding_fingerprint = ?
                    """,
                    (identity.value, embedding_fingerprint.value),
                ).fetchone()
                if dense is None:
                    dense = connection.execute(
                        """
                        SELECT * FROM dense_generations
                        WHERE corpus_id = ? AND embedding_fingerprint = ?
                        ORDER BY
                            CASE state WHEN 'BUILDING' THEN 0 WHEN 'FAILED' THEN 1 ELSE 2 END,
                            started_at DESC LIMIT 1
                        """,
                        (identity.value, embedding_fingerprint.value),
                    ).fetchone()
            if dense is None:
                dense_state = DenseState.ABSENT
            elif dense["state"] == "READY" and dense["source_revision"] == corpus["current_revision"] and dense["source_digest"] == corpus["content_digest"]:
                dense_state = DenseState.READY
            elif dense["state"] == "BUILDING":
                dense_state = DenseState.BUILDING
            elif dense["state"] == "FAILED":
                dense_state = DenseState.FAILED
            else:
                dense_state = DenseState.STALE
            return IndexStatus(
                corpus_id=identity.value,
                corpus_kind=identity.kind,
                vault_path=identity.vault_path,
                sparse_state=SparseState(corpus["sparse_state"]),
                dense_state=dense_state,
                revision=corpus["current_revision"],
                content_digest=corpus["content_digest"],
                documents=documents,
                chunks=chunks,
                dense_generation=dense["id"] if dense is not None else None,
                dense_revision=dense["source_revision"] if dense is not None else None,
                dense_rows=dense["row_count"] if dense is not None else 0,
                dense_dimension=dense["dimension"] if dense is not None else 0,
                dense_bytes=dense["vector_bytes"] if dense is not None else 0,
                embedding_fingerprint=dense["embedding_fingerprint"] if dense is not None else None,
                error_code=dense["error_code"] if dense is not None else corpus["sparse_error"],
                reused_rows=dense["reused_rows"] if dense is not None else 0,
                embedded_rows=dense["embedded_rows"] if dense is not None else 0,
                last_success_at=dense["completed_at"] if dense is not None else None,
            )
        finally:
            connection.close()

    def generation_root(self, identity: CorpusIdentity) -> Path:
        return generation_root(self.index_home, identity.prefix)

    def claim_dense_job(
        self,
        spec: CorpusSpec,
        *,
        revision: int,
        embedding: EmbeddingFingerprint,
        total: int,
        lease_seconds: int = 120,
    ) -> DenseJob | None:
        identity = spec.corpus_identity
        owner = uuid4().hex
        now = _now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE jobs SET state='EXPIRED', updated_at=? WHERE state IN ('CLAIMED', 'BUILDING') AND lease_expires_at < datetime('now')",
                (now,),
            )
            existing = connection.execute(
                """
                SELECT job_id FROM jobs
                WHERE corpus_id = ? AND kind = 'dense' AND requested_revision = ?
                  AND embedding_fingerprint = ? AND state IN ('CLAIMED', 'BUILDING')
                """,
                (identity.value, revision, embedding.value),
            ).fetchone()
            if existing is not None:
                connection.rollback()
                return None
            job_id = uuid4().hex
            connection.execute(
                """
                INSERT INTO jobs(
                    job_id, corpus_id, kind, requested_revision, embedding_fingerprint,
                    state, owner_instance_id, owner_pid, owner_started_at,
                    lease_expires_at, heartbeat_at, total, created_at, updated_at
                ) VALUES (?, ?, 'dense', ?, ?, 'CLAIMED', ?, ?, ?, datetime('now', ?), ?, ?, ?, ?)
                """,
                (
                    job_id,
                    identity.value,
                    revision,
                    embedding.value,
                    owner,
                    os.getpid(),
                    now,
                    f"+{lease_seconds} seconds",
                    now,
                    total,
                    now,
                    now,
                ),
            )
            connection.commit()
            return DenseJob(job_id, identity.value, revision, embedding.value, owner)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def begin_generation(
        self,
        spec: CorpusSpec,
        *,
        generation_id: str | None = None,
        revision: int,
        source_digest: str,
        embedding: EmbeddingFingerprint,
        vector_file: str,
        row_count: int,
        dimension: int,
        reused_rows: int,
        embedded_rows: int,
    ) -> str:
        generation_id = generation_id or uuid4().hex
        connection = self._connect()
        try:
            connection.execute(
                """
                INSERT INTO dense_generations(
                    id, corpus_id, source_revision, source_digest, embedding_fingerprint,
                    state, vector_file, row_count, dimension, reused_rows, embedded_rows,
                    started_at
                ) VALUES (?, ?, ?, ?, ?, 'BUILDING', ?, ?, ?, ?, ?, ?)
                """,
                (
                    generation_id,
                    spec.corpus_identity.value,
                    revision,
                    source_digest,
                    embedding.value,
                    vector_file,
                    row_count,
                    dimension,
                    reused_rows,
                    embedded_rows,
                    _now(),
                ),
            )
            return generation_id
        finally:
            connection.close()

    def publish_generation(
        self,
        spec: CorpusSpec,
        *,
        generation_id: str,
        rows: tuple[tuple[int, str, str], ...],
        vector_sha256: str,
        vector_bytes: int,
        job: DenseJob | None,
    ) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            generation = connection.execute(
                "SELECT * FROM dense_generations WHERE id=? AND corpus_id=? AND state='BUILDING'",
                (generation_id, spec.corpus_identity.value),
            ).fetchone()
            corpus = connection.execute(
                "SELECT current_revision, content_digest FROM corpora WHERE id=?",
                (spec.corpus_identity.value,),
            ).fetchone()
            if generation is None or corpus is None:
                raise RuntimeError("generation publish target is no longer valid")
            if (
                generation["source_revision"] != corpus["current_revision"]
                or generation["source_digest"] != corpus["content_digest"]
            ):
                raise RuntimeError("generation source changed before publish")
            if job is not None:
                owned = connection.execute(
                    """
                    SELECT 1 FROM jobs
                    WHERE job_id=? AND owner_instance_id=?
                      AND state IN ('CLAIMED', 'BUILDING')
                    """,
                    (job.job_id, job.owner_instance_id),
                ).fetchone()
                if owned is None:
                    raise RuntimeError("dense build lease is no longer owned")
            for row_index, chunk_id, input_hash in rows:
                connection.execute(
                    "INSERT INTO dense_generation_rows(generation_id, row_index, chunk_id, embedding_input_hash) VALUES (?, ?, ?, ?)",
                    (generation_id, row_index, chunk_id, input_hash),
                )
            updated = connection.execute(
                """
                UPDATE dense_generations
                SET state='READY', vector_sha256=?, vector_bytes=?, completed_at=?
                WHERE id=? AND state='BUILDING'
                """,
                (vector_sha256, vector_bytes, _now(), generation_id),
            )
            if updated.rowcount != 1:
                raise RuntimeError("dense generation was not published")
            pointer = connection.execute(
                """
                SELECT active_generation_id FROM dense_generation_pointers
                WHERE corpus_id=? AND embedding_fingerprint=?
                """,
                (spec.corpus_identity.value, generation["embedding_fingerprint"]),
            ).fetchone()
            previous = pointer["active_generation_id"] if pointer is not None else None
            connection.execute(
                """
                DELETE FROM dense_generation_pointers
                WHERE corpus_id=? AND embedding_fingerprint<>?
                """,
                (spec.corpus_identity.value, generation["embedding_fingerprint"]),
            )
            connection.execute(
                """
                INSERT INTO dense_generation_pointers(
                    corpus_id, embedding_fingerprint, active_generation_id,
                    previous_generation_id, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(corpus_id, embedding_fingerprint) DO UPDATE SET
                    previous_generation_id=dense_generation_pointers.active_generation_id,
                    active_generation_id=excluded.active_generation_id,
                    updated_at=excluded.updated_at
                """,
                (
                    spec.corpus_identity.value,
                    generation["embedding_fingerprint"],
                    generation_id,
                    previous,
                    _now(),
                ),
            )
            connection.execute(
                """
                UPDATE dense_generations
                SET state='SUPERSEDED'
                WHERE corpus_id=? AND state='READY' AND id NOT IN (?, COALESCE(?, ''))
                """,
                (spec.corpus_identity.value, generation_id, previous),
            )
            if job is not None:
                connection.execute(
                    "UPDATE jobs SET state='COMPLETED', completed=total, updated_at=? WHERE job_id=? AND owner_instance_id=?",
                    (_now(), job.job_id, job.owner_instance_id),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def fail_generation(self, generation_id: str, error_code: str, job: DenseJob | None) -> None:
        connection = self._connect()
        try:
            connection.execute(
                "UPDATE dense_generations SET state='FAILED', error_code=?, completed_at=? WHERE id=?",
                (error_code, _now(), generation_id),
            )
            if job is not None:
                connection.execute(
                    "UPDATE jobs SET state='FAILED', error_code=?, updated_at=? WHERE job_id=?",
                    (error_code, _now(), job.job_id),
                )
        finally:
            connection.close()

    def ready_generation(
        self,
        spec: CorpusSpec,
        *,
        revision: int,
        source_digest: str,
        embedding: EmbeddingFingerprint,
    ) -> sqlite3.Row | None:
        connection = self._connect()
        try:
            return connection.execute(
                """
                SELECT g.* FROM dense_generation_pointers p
                JOIN dense_generations g ON g.id=p.active_generation_id
                WHERE p.corpus_id=? AND p.embedding_fingerprint=?
                  AND g.source_revision=? AND g.source_digest=?
                  AND g.state='READY'
                """,
                (spec.corpus_identity.value, embedding.value, revision, source_digest),
            ).fetchone()
        finally:
            connection.close()

    def generation_rows(self, generation_id: str) -> tuple[tuple[int, str, str], ...]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT row_index, chunk_id, embedding_input_hash FROM dense_generation_rows WHERE generation_id=? ORDER BY row_index",
                (generation_id,),
            ).fetchall()
            return tuple((row["row_index"], row["chunk_id"], row["embedding_input_hash"]) for row in rows)
        finally:
            connection.close()

    def compatible_rows(
        self,
        spec: CorpusSpec,
        embedding: EmbeddingFingerprint,
    ) -> dict[str, tuple[str, int, str | None]]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT r.embedding_input_hash, g.vector_file, g.vector_sha256, r.row_index
                FROM dense_generation_rows r JOIN dense_generations g ON g.id=r.generation_id
                WHERE g.corpus_id=? AND g.embedding_fingerprint=?
                  AND g.state IN ('READY', 'SUPERSEDED')
                ORDER BY g.completed_at DESC
                """,
                (spec.corpus_identity.value, embedding.value),
            ).fetchall()
            result: dict[str, tuple[str, int, str | None]] = {}
            for row in rows:
                result.setdefault(
                    row["embedding_input_hash"],
                    (row["vector_file"], row["row_index"], row["vector_sha256"]),
                )
            return result
        finally:
            connection.close()

    def verify(self, spec: CorpusSpec) -> tuple[str, ...]:
        identity = spec.corpus_identity
        connection = self._connect()
        try:
            errors = list(connection.execute("PRAGMA integrity_check").fetchall())
            problems = [str(row[0]) for row in errors if row[0] != "ok"]
            foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
            problems.extend(f"foreign_key:{row[0]}:{row[1]}" for row in foreign_keys)
            corpus = connection.execute("SELECT id FROM corpora WHERE id=?", (identity.value,)).fetchone()
            if corpus is None:
                return tuple(problems)
            if stat.S_IMODE(self.db_path.stat().st_mode) != 0o600:
                problems.append("catalog_permissions")
            for pointer in connection.execute(
                """
                SELECT p.*, active.state AS active_state, previous.state AS previous_state
                FROM dense_generation_pointers p
                LEFT JOIN dense_generations active ON active.id=p.active_generation_id
                LEFT JOIN dense_generations previous ON previous.id=p.previous_generation_id
                WHERE p.corpus_id=?
                """,
                (identity.value,),
            ):
                if pointer["active_generation_id"] and pointer["active_state"] != "READY":
                    problems.append("active_generation_invalid")
                if pointer["previous_generation_id"] and pointer["previous_state"] not in {
                    "READY",
                    "SUPERSEDED",
                }:
                    problems.append("previous_generation_invalid")
            for row in connection.execute(
                "SELECT id, vector_file, vector_sha256, vector_bytes, row_count, dimension FROM dense_generations WHERE corpus_id=? AND state IN ('READY', 'SUPERSEDED')",
                (identity.value,),
            ):
                path = self.generation_root(identity) / row["vector_file"]
                if not path.is_file() or path.is_symlink():
                    problems.append(f"missing_generation_file:{row['id']}")
                    continue
                if row["vector_bytes"] and path.stat().st_size != row["vector_bytes"]:
                    problems.append(f"generation_size_mismatch:{row['id']}")
                if row["vector_sha256"]:
                    digest = hashlib.sha256()
                    with path.open("rb") as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(block)
                    if digest.hexdigest() != row["vector_sha256"]:
                        problems.append(f"generation_checksum_mismatch:{row['id']}")
                row_count = connection.execute(
                    "SELECT COUNT(*) FROM dense_generation_rows WHERE generation_id=?",
                    (row["id"],),
                ).fetchone()[0]
                if row_count != row["row_count"]:
                    problems.append(f"generation_rows_mismatch:{row['id']}")
                try:
                    vectors = np.load(path, allow_pickle=False, mmap_mode="r")
                    if vectors.dtype != np.float32 or vectors.shape != (row["row_count"], row["dimension"]):
                        problems.append(f"generation_shape_mismatch:{row['id']}")
                    elif not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) <= 1e-12):
                        problems.append(f"generation_vector_invalid:{row['id']}")
                except (OSError, ValueError, EOFError):
                    problems.append(f"generation_unreadable:{row['id']}")
            return tuple(problems)
        finally:
            connection.close()

    def gc(
        self,
        spec: CorpusSpec,
        *,
        apply: bool = False,
        grace_days: int = 7,
    ) -> tuple[str, ...]:
        identity = spec.corpus_identity
        root = self.generation_root(identity)
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT g.id, g.vector_file, g.state
                FROM dense_generations g
                LEFT JOIN dense_generation_pointers p
                  ON p.corpus_id=g.corpus_id
                 AND p.embedding_fingerprint=g.embedding_fingerprint
                WHERE g.corpus_id=?
                  AND g.state IN ('FAILED', 'SUPERSEDED')
                  AND g.id NOT IN (
                    COALESCE(p.active_generation_id, ''),
                    COALESCE(p.previous_generation_id, '')
                  )
                  AND COALESCE(g.completed_at, g.started_at)
                    < datetime('now', ?)
                """,
                (identity.value, f"-{grace_days} days"),
            ).fetchall()
            known = {
                row["vector_file"]
                for row in connection.execute(
                    "SELECT vector_file FROM dense_generations WHERE corpus_id=?",
                    (identity.value,),
                )
            }
            orphan_files = (
                tuple(
                    path
                    for path in root.iterdir()
                    if path.is_file()
                    and path.name not in known
                    and datetime.fromtimestamp(path.stat().st_mtime, UTC)
                    < datetime.now(UTC) - timedelta(days=grace_days)
                )
                if root.is_dir()
                else ()
            )
            candidates = tuple(str(root / row["vector_file"]) for row in rows) + tuple(
                str(path) for path in orphan_files
            )
            if apply:
                for row in rows:
                    connection.execute("DELETE FROM dense_generations WHERE id=?", (row["id"],))
                    (root / row["vector_file"]).unlink(missing_ok=True)
                for path in orphan_files:
                    path.unlink(missing_ok=True)
                connection.commit()
            return candidates
        finally:
            connection.close()

    def rollback_generation(
        self,
        spec: CorpusSpec,
        *,
        embedding: EmbeddingFingerprint,
    ) -> str:
        identity = spec.corpus_identity
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            pointer = connection.execute(
                """
                SELECT p.*, g.source_revision, g.source_digest, g.state
                FROM dense_generation_pointers p
                JOIN dense_generations g ON g.id=p.previous_generation_id
                WHERE p.corpus_id=? AND p.embedding_fingerprint=?
                """,
                (identity.value, embedding.value),
            ).fetchone()
            corpus = connection.execute(
                "SELECT current_revision, content_digest FROM corpora WHERE id=?",
                (identity.value,),
            ).fetchone()
            if pointer is None or corpus is None:
                raise RuntimeError("no compatible previous generation")
            if (
                pointer["state"] not in {"READY", "SUPERSEDED"}
                or pointer["source_revision"] != corpus["current_revision"]
                or pointer["source_digest"] != corpus["content_digest"]
            ):
                raise RuntimeError("previous generation is incompatible with the current corpus")
            old_active = pointer["active_generation_id"]
            new_active = pointer["previous_generation_id"]
            connection.execute(
                """
                UPDATE dense_generation_pointers
                SET active_generation_id=?, previous_generation_id=?, updated_at=?
                WHERE corpus_id=? AND embedding_fingerprint=?
                """,
                (new_active, old_active, _now(), identity.value, embedding.value),
            )
            connection.execute(
                "UPDATE dense_generations SET state='READY' WHERE id=?",
                (new_active,),
            )
            if old_active:
                connection.execute(
                    "UPDATE dense_generations SET state='SUPERSEDED' WHERE id=?",
                    (old_active,),
                )
            connection.commit()
            return str(new_active)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def inspect(self, spec: CorpusSpec) -> dict[str, object]:
        identity = spec.corpus_identity
        connection = self._connect()
        try:
            generations = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT id, source_revision, source_digest,
                           embedding_fingerprint, state, vector_file,
                           vector_sha256, vector_bytes, row_count, dimension,
                           reused_rows, embedded_rows, started_at, completed_at,
                           error_code
                    FROM dense_generations
                    WHERE corpus_id=?
                    ORDER BY started_at DESC
                    """,
                    (identity.value,),
                )
            ]
            pointers = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT embedding_fingerprint, active_generation_id,
                           previous_generation_id, updated_at
                    FROM dense_generation_pointers
                    WHERE corpus_id=?
                    """,
                    (identity.value,),
                )
            ]
            jobs = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT job_id, kind, requested_revision, state, total,
                           completed, reused, retry_count, error_code,
                           created_at, updated_at
                    FROM jobs WHERE corpus_id=? ORDER BY created_at DESC
                    """,
                    (identity.value,),
                )
            ]
            return {
                "status": self.status(spec).as_dict(),
                "pointers": pointers,
                "generations": generations,
                "jobs": jobs,
            }
        finally:
            connection.close()
