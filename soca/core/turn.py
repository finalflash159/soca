from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from soca.core.guardrails import GuardrailEvent
from soca.core.usage import LLMUsage
from soca.knowledge import KnowledgeCitation
from soca.tools import ToolCall, ToolResult


class RuntimeRoute(Enum):
    BLOCKED = "blocked"
    TOOL_DIRECT = "tool_direct"
    KNOWLEDGE_DIRECT = "knowledge_direct"
    KNOWLEDGE_LLM = "knowledge_llm"
    MEMORY_LLM = "memory_llm"
    MEMORY_DIRECT = "memory_direct"
    OUT_OF_SCOPE = "out_of_scope"
    CLARIFICATION = "clarification"
    FREE_CHAT = "free_chat"
    # Backward-compatible alias for older tests/reports that imported the enum name.
    LLM_FALLBACK = "free_chat"


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
    memory_hits: tuple[Any, ...] = ()
    memory_mode: str = "blob"
    memory_degraded_reason: str = ""
    citations: tuple[KnowledgeCitation, ...] = ()
    used_tool: bool = False
    used_llm: bool = False
    blocked: bool = False
    stage_latencies_ms: dict[str, float] = field(default_factory=dict)
    tool_router_tier: str = "none"
    tool_router_reason: str = "no_match"
    disposition: str = "unresolved"
    selected_sources: tuple[str, ...] = ()
    router_scores: dict[str, float] = field(default_factory=dict)
    router_runner_up: str | None = None
    router_margin: float | None = None
    evidence_decisions: tuple[Any, ...] = ()
    evidence_bundle: Any | None = None
    answer_validation: Any | None = None
    answer_repair_attempted: bool = False
    answer_repair_succeeded: bool = False
    prompt_manifest: dict[str, Any] | None = None


@dataclass(frozen=True)
class RuntimeResult:
    response_text: str
    route: RuntimeRoute
    blocked: bool = False
    citations: tuple[KnowledgeCitation, ...] = ()
    trace: RuntimeTrace | None = None
    frame: TurnFrame | None = None
    llm_result: Any | None = None
    # Normalized LLM telemetry for both streaming and non-streaming routes
    # (``llm_result`` stays the raw object; ``usage`` is the unified view).
    usage: LLMUsage | None = None


RuntimeStreamEventType = Literal["token", "sentence", "result"]


@dataclass(frozen=True)
class RuntimeStreamEvent:
    """Incremental output from AssistantRuntime.stream_text_turn.

    - ``token``: a raw LLM token, useful for live on-screen display.
    - ``sentence``: a guardrail-passed chunk ready to be sent to TTS.
    - ``result``: the terminal event carrying the full RuntimeResult.
    """

    type: RuntimeStreamEventType
    text: str = ""
    result: RuntimeResult | None = None


def iter_workflow_events(
    source: RuntimeResult | Iterable[RuntimeStreamEvent],
    *,
    turn_id: str = "",
) -> Iterator[Any]:
    """Compatibility entry point for the shared blocking/streaming adapter."""
    from .workflow.legacy_adapter import iter_runtime_events

    yield from iter_runtime_events(source, turn_id=turn_id)
