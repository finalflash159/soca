from __future__ import annotations

from soca.core.runtime import RuntimeToolRouter
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

    def select(self, text: str, *, knowledge_limit: int) -> ToolCall | None:
        call = self._deterministic.select(text, knowledge_limit=knowledge_limit)
        if call is not None:
            self.last_tier = "deterministic"
            return call
        if self._semantic is not None:
            call = self._semantic.select(text, knowledge_limit=knowledge_limit)
            if call is not None:
                self.last_tier = "semantic"
                return call
        if self._llm_router is None:
            self.last_tier = "none"
            return None
        call = self._llm_router.select(text, knowledge_limit=knowledge_limit)
        self.last_tier = getattr(self._llm_router, "last_tier", "llm" if call else "none")
        return call
