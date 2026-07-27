from __future__ import annotations

from pathlib import Path

from soca.knowledge.index.persistence import load_index, manifest_path_for
from soca.knowledge.indexing.catalog import IndexCatalog
from soca.knowledge.indexing.identity import CorpusSpec


def v1_manifest_path(index_home: Path, vault_path: Path) -> Path:
    return manifest_path_for(index_home, vault_path)


def import_v1_manifest(
    catalog: IndexCatalog,
    spec: CorpusSpec,
    *,
    legacy_index_home: Path,
) -> bool:
    path = v1_manifest_path(legacy_index_home, spec.vault_path)
    index = load_index(path, expected_vault_path=spec.resolved_vault_path)
    if index is None:
        return False
    return catalog.import_sparse_index(spec, index)
