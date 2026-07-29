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
            return deterministic
        return LLMToolRouter(llm, tool_runtime, config=config, fallback=deterministic)

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
    if semantic_router is None and config.semantic.enabled and embedding_model is None:
        # Text's semantic default is explicitly offline-safe: without an
        # embedder, degrade to Tier 0 instead of paying a second LLM call for
        # every ordinary chat turn.
        return deterministic
    llm_router = (
        LLMToolRouter(llm, tool_runtime, config=config, fallback=None)
        if llm is not None and (not voice or config.enabled_in_voice)
        else None
    )
    if semantic_router is None and llm_router is None:
        return deterministic
    return CascadeToolRouter(deterministic, semantic_router, llm_router)


__all__ = ["build_runtime_tool_router"]
