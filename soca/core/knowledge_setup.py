from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from soca.knowledge import KnowledgeCatalog, KnowledgeContextBuilder, KnowledgeSource
from soca.knowledge.catalog import CatalogIndexProvider
from soca.knowledge.factory import RetrievalConfig, build_knowledge_source
from soca.knowledge.relevance import RelevancePolicy
from soca.tools import (
    KnowledgeInspectTool,
    KnowledgeReadTool,
    KnowledgeSearchTool,
)


@dataclass(frozen=True)
class KnowledgeRuntimeSetup:
    source: KnowledgeSource
    catalog: KnowledgeCatalog
    builder: KnowledgeContextBuilder
    inspect_tool: KnowledgeInspectTool
    search_tool: KnowledgeSearchTool
    read_tool: KnowledgeReadTool
    status: str


def build_knowledge_runtime_setup(
    vault: Path,
    *,
    knowledge_limit: int,
    retrieval_config: RetrievalConfig | None = None,
    index_home: Path | None = None,
) -> KnowledgeRuntimeSetup:
    resolved_config = retrieval_config or RetrievalConfig()
    source = build_knowledge_source(
        vault,
        config=resolved_config,
        index_home=index_home,
    )
    try:
        effective_mode = str(getattr(source, "retrieval_mode", resolved_config.mode))
        relevance_policy = RelevancePolicy.for_retrieval_mode(effective_mode)
        catalog = KnowledgeCatalog(cast(CatalogIndexProvider, source))
        return KnowledgeRuntimeSetup(
            source=source,
            catalog=catalog,
            builder=KnowledgeContextBuilder(
                source,
                max_hits=knowledge_limit,
                relevance_policy=relevance_policy,
                catalog=catalog,
            ),
            inspect_tool=KnowledgeInspectTool(catalog),
            search_tool=KnowledgeSearchTool(
                source,
                max_limit=knowledge_limit,
                relevance_policy=relevance_policy,
            ),
            read_tool=KnowledgeReadTool(source),
            status=f"enabled:{effective_mode}",
        )
    except Exception:
        close = getattr(source, "close", None)
        if callable(close):
            try:
                close()
            except Exception as cleanup_exc:  # noqa: BLE001 - preserve cleanup failure
                raise RuntimeError("knowledge runtime setup cleanup failed") from cleanup_exc
        raise
