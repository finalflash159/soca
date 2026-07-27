from __future__ import annotations

from dataclasses import dataclass

from soca.knowledge.index.models import VaultIndex
from soca.knowledge.indexing.catalog import IndexCatalog
from soca.knowledge.indexing.generations import DenseBuildReport, DenseGenerationBuilder
from soca.knowledge.indexing.identity import CorpusSpec
from soca.knowledge.retrievers.dense import DenseIndex, EmbeddingModel


@dataclass(frozen=True)
class DenseWorkerResult:
    index: DenseIndex
    report: DenseBuildReport


class DenseWorker:
    """Synchronous worker seam used by CLI and future background engines."""

    def __init__(self, catalog: IndexCatalog) -> None:
        self.builder = DenseGenerationBuilder(catalog)

    def run(
        self,
        spec: CorpusSpec,
        *,
        sparse_index: VaultIndex,
        revision: int,
        model: EmbeddingModel,
    ) -> DenseWorkerResult:
        index, report = self.builder.build(
            spec,
            index=sparse_index,
            revision=revision,
            model=model,
        )
        return DenseWorkerResult(index=index, report=report)
