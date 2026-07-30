from __future__ import annotations

from collections.abc import Callable
from typing import cast

from soca.core.runtime import RuntimeToolRouter
from soca.core.tool_routing import EvidenceCompletionDecision, ToolRouterDecision
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

    def set_context(self, *, turn_context: str = "") -> None:
        setter = getattr(self._llm_router, "set_context", None)
        if callable(setter):
            setter(turn_context=turn_context)

    def refine(
        self,
        text: str,
        *,
        observation: str,
        knowledge_limit: int,
    ) -> ToolCall | None:
        refiner = getattr(self._llm_router, "refine", None)
        if not callable(refiner):
            return None
        typed_refiner = cast(Callable[..., ToolCall | None], refiner)
        return typed_refiner(
            text,
            observation=observation,
            knowledge_limit=knowledge_limit,
        )

    def assess_evidence(
        self,
        text: str,
        *,
        observation: str,
        knowledge_limit: int,
    ) -> EvidenceCompletionDecision | None:
        assessor = getattr(self._llm_router, "assess_evidence", None)
        if not callable(assessor):
            return None
        typed_assessor = cast(Callable[..., EvidenceCompletionDecision], assessor)
        return typed_assessor(
            text,
            observation=observation,
            knowledge_limit=knowledge_limit,
        )

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
            # Semantic examples are shadow telemetry only.  They may not
            # terminate a turn: the model router must see the vault manifest,
            # goal and conversation context before selecting an action.
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
