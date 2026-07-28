from __future__ import annotations

from soca.core.runtime import RuntimeToolRouter
from soca.core.tool_routing import ToolRouterDecision
from soca.tools import ToolCall


class CascadeToolRouter:
    def __init__(
        self,
        deterministic: RuntimeToolRouter,
        semantic: RuntimeToolRouter | None,
        llm_router: RuntimeToolRouter | None,
    ) -> None:
        self._deterministic = deterministic
        self._semantic = semantic
        self._llm_router = llm_router
        self.last_tier = "none"
        self.last_decision = ToolRouterDecision()

    def select(self, text: str, *, knowledge_limit: int) -> ToolCall | None:
        call = self._deterministic.select(text, knowledge_limit=knowledge_limit)
        deterministic_decision = getattr(
            self._deterministic, "last_decision", ToolRouterDecision()
        )
        if call is not None:
            self.last_tier = "deterministic"
            self.last_decision = deterministic_decision
            return call
        if self._semantic is not None:
            call = self._semantic.select(text, knowledge_limit=knowledge_limit)
            semantic_decision = getattr(self._semantic, "last_decision", ToolRouterDecision())
            if call is not None:
                self.last_tier = "semantic"
                self.last_decision = semantic_decision
                return call
            if semantic_decision.disposition in {
                "retrieval_request",
                "smalltalk",
                "out_of_scope",
                "unresolved",
            }:
                self.last_tier = "semantic"
                self.last_decision = semantic_decision
                return None
        if self._llm_router is None:
            self.last_tier = "none"
            self.last_decision = getattr(
                self._semantic, "last_decision", deterministic_decision
            )
            return None
        call = self._llm_router.select(text, knowledge_limit=knowledge_limit)
        self.last_tier = getattr(self._llm_router, "last_tier", "llm" if call else "none")
        self.last_decision = getattr(
            self._llm_router, "last_decision", ToolRouterDecision(call=call, reason="llm_match" if call else "llm_none")
        )
        return call
