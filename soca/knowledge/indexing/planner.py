from __future__ import annotations

from dataclasses import dataclass

from soca.knowledge.index.models import MarkdownChunk
from soca.knowledge.indexing.identity import EmbeddingFingerprint, embedding_input_hash


@dataclass(frozen=True)
class DenseBuildPlan:
    chunks: tuple[MarkdownChunk, ...]
    input_hashes: tuple[str, ...]
    reuse_positions: tuple[int, ...]
    embed_positions: tuple[int, ...]

    @property
    def reused_rows(self) -> int:
        return len(self.reuse_positions)

    @property
    def embedded_rows(self) -> int:
        return len(self.embed_positions)


def plan_dense_build(
    chunks: tuple[MarkdownChunk, ...],
    fingerprint: EmbeddingFingerprint,
    reusable_hashes: set[str],
) -> DenseBuildPlan:
    hashes = tuple(embedding_input_hash(fingerprint, chunk.text) for chunk in chunks)
    reuse = tuple(index for index, value in enumerate(hashes) if value in reusable_hashes)
    embed = tuple(index for index, value in enumerate(hashes) if value not in reusable_hashes)
    return DenseBuildPlan(chunks, hashes, reuse, embed)
