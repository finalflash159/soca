from __future__ import annotations

import sqlite3
import stat
from pathlib import Path

from soca.knowledge.index.persistence import ensure_private_directory

SCHEMA_VERSION = 3
PRIVATE_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR


def _ensure_private_file(path: Path) -> None:
    """Keep the SQLite catalog and its WAL sidecars owner-only."""
    try:
        path.chmod(PRIVATE_FILE_MODE)
    except FileNotFoundError:
        pass


def connect_catalog(path: Path) -> sqlite3.Connection:
    sqlite_path = path if str(path) == ":memory:" else path.expanduser().resolve()
    if str(sqlite_path) != ":memory:":
        ensure_private_directory(sqlite_path.parent)
        _ensure_private_file(sqlite_path)
    connection = sqlite3.connect(sqlite_path, timeout=5.0, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    if str(sqlite_path) != ":memory:":
        try:
            connection.execute("PRAGMA journal_mode = WAL")
        except sqlite3.DatabaseError:
            connection.execute("PRAGMA journal_mode = DELETE")
    connection.execute("PRAGMA synchronous = FULL")
    ensure_schema(connection)
    if str(sqlite_path) != ":memory:":
        _ensure_private_file(sqlite_path)
        _ensure_private_file(Path(f"{sqlite_path}-wal"))
        _ensure_private_file(Path(f"{sqlite_path}-shm"))
    return connection


def ensure_schema(connection: sqlite3.Connection) -> None:
    current = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if current > SCHEMA_VERSION:
        raise RuntimeError(f"unsupported index catalog schema: {current}")
    if current == 0:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS corpora (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL CHECK(kind IN ('knowledge', 'memory')),
                vault_path TEXT NOT NULL,
                policy_json TEXT NOT NULL,
                chunker_fingerprint TEXT NOT NULL,
                current_revision INTEGER NOT NULL DEFAULT 0,
                content_digest TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                sparse_state TEXT NOT NULL DEFAULT 'absent',
                sparse_error TEXT
            );
            CREATE TABLE IF NOT EXISTS files (
                corpus_id TEXT NOT NULL,
                path TEXT NOT NULL,
                mtime_ns INTEGER NOT NULL,
                ctime_ns INTEGER NOT NULL,
                size INTEGER NOT NULL,
                inode INTEGER NOT NULL,
                content_sha256 TEXT NOT NULL,
                title TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                frontmatter_json TEXT NOT NULL,
                text TEXT NOT NULL,
                revision INTEGER NOT NULL,
                PRIMARY KEY(corpus_id, path),
                FOREIGN KEY(corpus_id) REFERENCES corpora(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS chunks (
                corpus_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                document_path TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                title TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                text TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                revision INTEGER NOT NULL,
                PRIMARY KEY(corpus_id, chunk_id),
                FOREIGN KEY(corpus_id) REFERENCES corpora(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS chunks_document_order
                ON chunks(corpus_id, document_path, ordinal);
            CREATE INDEX IF NOT EXISTS chunks_revision
                ON chunks(corpus_id, revision);
            CREATE INDEX IF NOT EXISTS chunks_content_sha
                ON chunks(corpus_id, content_sha256);
            CREATE TABLE IF NOT EXISTS embedding_models (
                fingerprint TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                install_state TEXT NOT NULL,
                artifact_path TEXT,
                artifact_digest TEXT,
                verified_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dense_generations (
                id TEXT PRIMARY KEY,
                corpus_id TEXT NOT NULL,
                source_revision INTEGER NOT NULL,
                source_digest TEXT NOT NULL,
                embedding_fingerprint TEXT NOT NULL,
                state TEXT NOT NULL,
                vector_file TEXT NOT NULL,
                vector_sha256 TEXT,
                vector_bytes INTEGER NOT NULL DEFAULT 0,
                row_count INTEGER NOT NULL,
                dimension INTEGER NOT NULL,
                reused_rows INTEGER NOT NULL DEFAULT 0,
                embedded_rows INTEGER NOT NULL DEFAULT 0,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                error_code TEXT,
                FOREIGN KEY(corpus_id) REFERENCES corpora(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS dense_generations_current
                ON dense_generations(corpus_id, source_revision, state);
            CREATE TABLE IF NOT EXISTS dense_generation_rows (
                generation_id TEXT NOT NULL,
                row_index INTEGER NOT NULL,
                chunk_id TEXT NOT NULL,
                embedding_input_hash TEXT NOT NULL,
                PRIMARY KEY(generation_id, row_index),
                FOREIGN KEY(generation_id) REFERENCES dense_generations(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS dense_rows_input
                ON dense_generation_rows(embedding_input_hash);
            CREATE TABLE IF NOT EXISTS dense_search_artifacts (
                generation_id TEXT NOT NULL,
                backend_fingerprint TEXT NOT NULL,
                state TEXT NOT NULL,
                artifact_file TEXT,
                artifact_sha256 TEXT,
                artifact_bytes INTEGER NOT NULL DEFAULT 0,
                build_ms REAL,
                created_at TEXT NOT NULL,
                error_code TEXT,
                PRIMARY KEY(generation_id, backend_fingerprint),
                FOREIGN KEY(generation_id) REFERENCES dense_generations(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS dense_generation_pointers (
                corpus_id TEXT NOT NULL,
                embedding_fingerprint TEXT NOT NULL,
                active_generation_id TEXT,
                previous_generation_id TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(corpus_id, embedding_fingerprint),
                FOREIGN KEY(corpus_id) REFERENCES corpora(id) ON DELETE CASCADE,
                FOREIGN KEY(active_generation_id) REFERENCES dense_generations(id),
                FOREIGN KEY(previous_generation_id) REFERENCES dense_generations(id)
            );
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                corpus_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                requested_revision INTEGER NOT NULL,
                embedding_fingerprint TEXT,
                state TEXT NOT NULL,
                owner_instance_id TEXT,
                owner_pid INTEGER,
                owner_started_at TEXT,
                lease_expires_at TEXT,
                heartbeat_at TEXT,
                total INTEGER NOT NULL DEFAULT 0,
                completed INTEGER NOT NULL DEFAULT 0,
                reused INTEGER NOT NULL DEFAULT 0,
                retry_count INTEGER NOT NULL DEFAULT 0,
                error_code TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(corpus_id) REFERENCES corpora(id) ON DELETE CASCADE
            );
            PRAGMA user_version = 3;
            """
        )
    elif current == 1:
        raise RuntimeError("index catalog schema v1 cannot be upgraded implicitly")
    elif current == 2:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS dense_generation_pointers (
                corpus_id TEXT NOT NULL,
                embedding_fingerprint TEXT NOT NULL,
                active_generation_id TEXT,
                previous_generation_id TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(corpus_id, embedding_fingerprint),
                FOREIGN KEY(corpus_id) REFERENCES corpora(id) ON DELETE CASCADE,
                FOREIGN KEY(active_generation_id) REFERENCES dense_generations(id),
                FOREIGN KEY(previous_generation_id) REFERENCES dense_generations(id)
            );
            INSERT OR IGNORE INTO dense_generation_pointers(
                corpus_id, embedding_fingerprint, active_generation_id, updated_at
            )
            SELECT corpus_id, embedding_fingerprint, id, COALESCE(completed_at, started_at)
            FROM (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY corpus_id, embedding_fingerprint
                        ORDER BY completed_at DESC, started_at DESC
                    ) AS position
                FROM dense_generations
                WHERE state='READY'
            )
            WHERE position=1;
            PRAGMA user_version = 3;
            """
        )


def catalog_path(index_home: Path) -> Path:
    return index_home.expanduser() / "v2" / "index.sqlite3"


def generation_root(index_home: Path, corpus_prefix: str) -> Path:
    return index_home.expanduser() / "v2" / "generations" / corpus_prefix
