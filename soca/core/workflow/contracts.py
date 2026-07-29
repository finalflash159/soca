from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal


class GoalStatus(StrEnum):
    ACTIVE = "active"
    ACHIEVED = "achieved"
    ABANDONED = "abandoned"


class TurnState(StrEnum):
    RECEIVED = "received"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    AUTHORIZING = "authorizing"
    EXECUTING = "executing"
    OBSERVING = "observing"
    SYNTHESIZING = "synthesizing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeStatus(StrEnum):
    CONTINUE = "continue"
    WAITING = "waiting"
    RETRY = "retry"
    TERMINAL = "terminal"


class TerminalStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class GoalContract:
    statement: str
    goal_id: str = ""
    source: Literal["text", "voice", "follow_up"] = "text"
    success_criteria: tuple[str, ...] = ()
    status: GoalStatus = GoalStatus.ACTIVE
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        statement = self.statement.strip()
        if not statement:
            raise ValueError("goal statement must not be empty")
        if any(not criterion.strip() for criterion in self.success_criteria):
            raise ValueError("goal success criteria must not contain empty values")
        object.__setattr__(self, "statement", statement)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class TurnBudget:
    max_transitions: int = 12
    max_tool_calls: int = 4
    max_model_calls: int = 4
    max_retries: int = 1
    max_elapsed_ms: int | None = None

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value is None and name == "max_elapsed_ms":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class NodeOutcome:
    node: str
    status: NodeStatus
    state: TurnState
    output: Any = None
    reason: str = ""
    retryable: bool = False


@dataclass(frozen=True)
class TerminalOutcome:
    status: TerminalStatus
    response_text: str = ""
    route: str | None = None
    error_code: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "response_text", self.response_text.strip())
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
