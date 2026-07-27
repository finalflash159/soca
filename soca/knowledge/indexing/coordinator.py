from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from soca.knowledge.index.models import VaultIndex
from soca.knowledge.indexing.catalog import IndexCatalog, SparseSyncResult
from soca.knowledge.indexing.generations import DenseBuildReport, DenseGenerationBuilder
from soca.knowledge.indexing.identity import (
    ChunkerFingerprint,
    CorpusSpec,
    EmbeddingFingerprint,
    embedding_fingerprint_for,
)
from soca.knowledge.indexing.migration_v1 import import_v1_manifest
from soca.knowledge.indexing.status import DenseState, IndexStatus
from soca.knowledge.retrievers.dense import DenseIndex, EmbeddingModel


@dataclass(frozen=True)
class IndexSnapshot:
    corpus_id: str
    revision: int
    content_digest: str
    sparse_index: VaultIndex
    dense_index: DenseIndex | None
    dense_state: DenseState
    embedding_fingerprint: EmbeddingFingerprint | None


@dataclass(frozen=True)
class BuildReport:
    sparse: SparseSyncResult
    dense: DenseBuildReport | None


class IndexCoordinator:
    """Coordinates sparse revisions and immutable dense generations.

    ``snapshot`` never calls ``embed_documents``. Document encoding is only
    reachable through ``build_blocking`` (CLI/worker path).
    """

    def __init__(
        self,
        reader: object,
        *,
        spec: CorpusSpec,
        index_home: Path,
        model: EmbeddingModel | None = None,
        chunker: ChunkerFingerprint | None = None,
    ) -> None:
        self.reader = reader
        self.spec = spec
        self.catalog = IndexCatalog(index_home)
        self.builder = DenseGenerationBuilder(self.catalog)
        self.model = model
        self.chunker = chunker or ChunkerFingerprint()
        self._lock = RLock()

    def sync_sparse(self, *, verify_content: bool = False) -> SparseSyncResult:
        with self._lock:
            if self.catalog.sparse_index(self.spec.corpus_identity) is None:
                import_v1_manifest(
                    self.catalog,
                    self.spec,
                    legacy_index_home=self.catalog.index_home,
                )
            return self.catalog.sync_sparse(
                self.spec,
                self.reader,  # type: ignore[arg-type]
                chunker=self.chunker,
                verify_content=verify_content,
            )

    def snapshot(self) -> IndexSnapshot:
        with self._lock:
            sparse = self.sync_sparse()
            dense: DenseIndex | None = None
            fingerprint: EmbeddingFingerprint | None = None
            if self.model is not None and sparse.index.chunks:
                fingerprint = embedding_fingerprint_for(self.model)
                dense = self.builder.load_ready(
                    self.spec,
                    index=sparse.index,
                    revision=sparse.revision,
                    model=self.model,
                )
            if dense is not None:
                dense_state = DenseState.READY
            elif self.model is None:
                dense_state = DenseState.MODEL_MISSING
            else:
                status = self.catalog.status(self.spec, embedding_fingerprint=fingerprint)
                dense_state = (
                    DenseState.STALE
                    if status.dense_state not in {DenseState.ABSENT, DenseState.MODEL_MISSING}
                    else status.dense_state
                )
            return IndexSnapshot(
                corpus_id=self.spec.corpus_identity.value,
                revision=sparse.revision,
                content_digest=sparse.index.content_digest,
                sparse_index=sparse.index,
                dense_index=dense,
                dense_state=dense_state,
                embedding_fingerprint=fingerprint,
            )

    def build_blocking(
        self,
        *,
        dense: bool = True,
        verify_content: bool = False,
        force_dense: bool = False,
    ) -> BuildReport:
        with self._lock:
            sparse = self.sync_sparse(verify_content=verify_content)
            dense_report = None
            if dense:
                _, dense_report = self.builder.build(
                    self.spec,
                    index=sparse.index,
                    revision=sparse.revision,
                    model=self.model,
                    force=force_dense,
                )
            return BuildReport(sparse=sparse, dense=dense_report)

    def status(self) -> IndexStatus:
        fingerprint = embedding_fingerprint_for(self.model) if self.model is not None else None
        return self.catalog.status(self.spec, embedding_fingerprint=fingerprint)

    def verify(self) -> tuple[str, ...]:
        return self.catalog.verify(self.spec)

    def gc(self, *, apply: bool = False) -> tuple[str, ...]:
        return self.catalog.gc(self.spec, apply=apply)

    def request_sync(self, reason: str = "manual") -> SparseSyncResult:
        del reason
        return self.sync_sparse()
