from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class SparseState(StrEnum):
    ABSENT = "absent"
    SCANNING = "scanning"
    READY = "ready"
    STALE = "stale"
    FAILED = "failed"


class DenseState(StrEnum):
    MODEL_MISSING = "model_missing"
    ABSENT = "absent"
    QUEUED = "queued"
    BUILDING = "building"
    READY = "ready"
    STALE = "stale"
    INCOMPATIBLE = "incompatible"
    FAILED = "failed"


class IndexErrorCode(StrEnum):
    MODEL_MISSING = "model_missing"
    ARTIFACT_MISMATCH = "artifact_mismatch"
    ENCODE_FAILED = "encode_failed"
    DISK_FULL = "disk_full"
    GENERATION_CORRUPT = "generation_corrupt"
    LEASE_LOST = "lease_lost"
    CATALOG_CORRUPT = "catalog_corrupt"


@dataclass(frozen=True)
class IndexStatus:
    corpus_id: str
    corpus_kind: str
    vault_path: str
    sparse_state: SparseState
    dense_state: DenseState
    revision: int
    content_digest: str | None
    documents: int
    chunks: int
    dense_generation: str | None = None
    dense_revision: int | None = None
    dense_rows: int = 0
    dense_dimension: int = 0
    dense_bytes: int = 0
    embedding_fingerprint: str | None = None
    stale_reason: str | None = None
    error_code: str | None = None
    reused_rows: int = 0
    embedded_rows: int = 0
    last_success_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "corpus_kind": self.corpus_kind,
            "vault_path": self.vault_path,
            "sparse_state": str(self.sparse_state),
            "dense_state": str(self.dense_state),
            "revision": self.revision,
            "content_digest": self.content_digest,
            "documents": self.documents,
            "chunks": self.chunks,
            "dense_generation": self.dense_generation,
            "dense_revision": self.dense_revision,
            "dense_rows": self.dense_rows,
            "dense_dimension": self.dense_dimension,
            "dense_bytes": self.dense_bytes,
            "embedding_fingerprint": self.embedding_fingerprint,
            "stale_reason": self.stale_reason,
            "error_code": self.error_code,
            "reused_rows": self.reused_rows,
            "embedded_rows": self.embedded_rows,
            "last_success_at": self.last_success_at,
        }
