from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from soca.knowledge import KnowledgeContextBuilder, KnowledgeSource
from soca.knowledge.factory import RetrievalConfig, build_knowledge_source
from soca.tools import KnowledgeReadTool, KnowledgeSearchTool


@dataclass(frozen=True)
class KnowledgeRuntimeSetup:
    source: KnowledgeSource
    builder: KnowledgeContextBuilder
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
    source = build_knowledge_source(
        vault,
        config=retrieval_config,
        index_home=index_home,
    )
    return KnowledgeRuntimeSetup(
        source=source,
        builder=KnowledgeContextBuilder(source, max_hits=knowledge_limit),
        search_tool=KnowledgeSearchTool(source, max_limit=knowledge_limit),
        read_tool=KnowledgeReadTool(source),
        status="enabled",
    )
