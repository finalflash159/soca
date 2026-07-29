from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from soca.knowledge import KnowledgeContextBuilder, KnowledgeSource
from soca.knowledge.factory import RetrievalConfig, build_knowledge_source
from soca.knowledge.relevance import RelevancePolicy
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
    resolved_config = retrieval_config or RetrievalConfig()
    source = build_knowledge_source(
        vault,
        config=resolved_config,
        index_home=index_home,
    )
    relevance_policy = RelevancePolicy.for_retrieval_mode(resolved_config.mode)
    return KnowledgeRuntimeSetup(
        source=source,
        builder=KnowledgeContextBuilder(
            source,
            max_hits=knowledge_limit,
            relevance_policy=relevance_policy,
        ),
        search_tool=KnowledgeSearchTool(source, max_limit=knowledge_limit),
        read_tool=KnowledgeReadTool(source),
        status="enabled",
    )
