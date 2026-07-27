"""Transactional knowledge-index lifecycle primitives.

The package deliberately keeps the SQLite catalog and the vector matrix separate:
SQLite is the source of truth for corpus/chunk/generation metadata, while a
validated immutable ``.npy`` file is the source of truth for dense vectors.
"""

__all__ = [
    "CorpusIdentity",
    "CorpusSpec",
    "IndexCatalog",
    "IndexCoordinator",
    "IndexStatus",
]


def __getattr__(name: str):
    if name == "CorpusIdentity" or name == "CorpusSpec":
        from soca.knowledge.indexing.identity import CorpusIdentity, CorpusSpec

        return CorpusIdentity if name == "CorpusIdentity" else CorpusSpec
    if name == "IndexCatalog":
        from soca.knowledge.indexing.catalog import IndexCatalog

        return IndexCatalog
    if name == "IndexCoordinator":
        from soca.knowledge.indexing.coordinator import IndexCoordinator

        return IndexCoordinator
    if name == "IndexStatus":
        from soca.knowledge.indexing.status import IndexStatus

        return IndexStatus
    raise AttributeError(name)
