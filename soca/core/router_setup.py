from __future__ import annotations

from dataclasses import replace

from soca.core.llm_tool_router import LLMToolRouter
from soca.core.router_cascade import CascadeToolRouter
from soca.core.runtime import DefaultRuntimeToolRouter, RuntimeToolRouter
from soca.core.semantic_tool_router import build_semantic_router
from soca.core.tool_routing import ToolRouterConfig
from soca.knowledge.retrievers.dense import EmbeddingModel
from soca.llm import LLMEngine
from soca.tools import ToolRuntime


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

    semantic_config = (
        replace(config.semantic, enabled=False)
        if voice and not config.semantic_enabled_in_voice
        else config.semantic
    )
    semantic_router = build_semantic_router(
        tool_runtime=tool_runtime,
        config=semantic_config,
        embedding_model=embedding_model,
    )
    llm_router = (
        LLMToolRouter(llm, tool_runtime, config=config, fallback=None)
        if llm is not None and (not voice or config.enabled_in_voice)
        else None
    )
    if semantic_router is None and llm_router is None:
        return deterministic
    return CascadeToolRouter(deterministic, semantic_router, llm_router)


__all__ = ["build_runtime_tool_router"]
