from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from soca.core.guardrails import GuardrailEvent
from soca.knowledge import KnowledgeCitation
from soca.tools import ToolCall, ToolResult


class RuntimeRoute(Enum):
    BLOCKED = "blocked"
    TOOL_DIRECT = "tool_direct"
    KNOWLEDGE_DIRECT = "knowledge_direct"
    KNOWLEDGE_LLM = "knowledge_llm"
    LLM_FALLBACK = "llm_fallback"


@dataclass(frozen=True)
class TurnFrame:
    text: str
    source: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeTrace:
    route: RuntimeRoute
    guardrail_events: tuple[GuardrailEvent, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    knowledge_hits: tuple[Any, ...] = ()
    citations: tuple[KnowledgeCitation, ...] = ()
    used_tool: bool = False
    used_llm: bool = False
    blocked: bool = False
    stage_latencies_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeResult:
    response_text: str
    route: RuntimeRoute
    blocked: bool = False
    citations: tuple[KnowledgeCitation, ...] = ()
    trace: RuntimeTrace | None = None
    frame: TurnFrame | None = None
    llm_result: Any | None = None

