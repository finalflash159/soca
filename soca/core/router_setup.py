from __future__ import annotations

import logging

from soca.core.llm_tool_router import LLMToolRouter
from soca.core.router_cascade import CascadeToolRouter
from soca.core.runtime import DefaultRuntimeToolRouter, RuntimeToolRouter
from soca.core.semantic_turn_router import build_semantic_turn_router
from soca.core.tool_routing import ToolRouterConfig
from soca.knowledge.retrievers.dense import EmbeddingModel
from soca.llm import LLMEngine
from soca.tools import ToolRuntime

LOGGER = logging.getLogger(__name__)


def build_runtime_tool_router(
    *,
    llm: LLMEngine | None,
    tool_runtime: ToolRuntime,
    deterministic: DefaultRuntimeToolRouter,
    config: ToolRouterConfig,
    embedding_model: EmbeddingModel | None,
    voice: bool,
) -> RuntimeToolRouter:
    if config.mode == "deterministic":
        return deterministic
    if config.mode == "llm":
        if llm is None or (voice and not config.enabled_in_voice):
            raise RuntimeError("llm_tool_router_unavailable")
        return LLMToolRouter(llm, tool_runtime, config=config)

    # Capability routing is surface-independent: the ASR transcript enters
    # the same semantic policy as text.  Only the optional LLM-router tier has
    # a separate voice privacy/latency gate.
    semantic_config = config.semantic
    try:
        semantic_router = build_semantic_turn_router(
            tool_runtime=tool_runtime,
            config=semantic_config,
            embedding_model=embedding_model,
        )
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
        LOGGER.warning("Semantic tool router unavailable; using lower tiers (%s)", type(exc).__name__)
        semantic_router = None
    llm_router = None
    if llm is not None and (not voice or config.enabled_in_voice):
        # Deterministic and semantic tiers have already had their chance. The
        # LLM tier is a bounded capability classifier, not an answer fallback.
        llm_router = LLMToolRouter(llm, tool_runtime, config=config)
    if semantic_router is None and llm_router is None:
        return deterministic
    return CascadeToolRouter(deterministic, semantic_router, llm_router)


__all__ = ["build_runtime_tool_router"]
