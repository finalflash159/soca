from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from .contracts import TerminalOutcome, TurnNode, TurnSource
from .errors import DuplicateTerminalError

PROTOCOL_VERSION = 2


class EventType(StrEnum):
    TURN_STARTED = "turn_started"
    GOAL_RESOLVED = "goal_resolved"
    STEP_STARTED = "step_started"
    STEP_PROGRESS = "step_progress"
    STEP_COMPLETED = "step_completed"
    PUBLIC_UPDATE = "public_update"
    ANSWER_DELTA = "answer_delta"
    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_COMPLETED = "verification_completed"
    TURN_TERMINAL = "turn_terminal"


class EventStatus(StrEnum):
    STARTED = "started"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class WorkflowEvent:
    event: EventType
    session_id: str
    run_id: str
    goal_id: str
    sequence: int
    surface: TurnSource
    timestamp: str
    node: TurnNode
    status: EventStatus
    payload: Mapping[str, Any] = field(default_factory=dict)
    protocol_version: int = PROTOCOL_VERSION

    @property
    def terminal(self) -> bool:
        return self.event is EventType.TURN_TERMINAL


class WorkflowEventStream:
    def __init__(
        self,
        *,
        session_id: str,
        run_id: str,
        goal_id: str,
        surface: TurnSource,
    ) -> None:
        for name, value in {
            "session_id": session_id,
            "run_id": run_id,
            "goal_id": goal_id,
        }.items():
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        self.session_id: str = session_id
        self.run_id: str = run_id
        self.goal_id: str = goal_id
        self.surface: TurnSource = surface
        self._events: list[WorkflowEvent] = []
        self._terminal: TerminalOutcome | None = None

    @property
    def terminal_outcome(self) -> TerminalOutcome | None:
        return self._terminal

    def emit(
        self,
        event: EventType,
        node: TurnNode,
        *,
        status: EventStatus = EventStatus.ACTIVE,
        payload: Mapping[str, Any] | None = None,
    ) -> WorkflowEvent:
        if event is EventType.TURN_TERMINAL:
            raise ValueError("use emit_terminal for terminal events")
        if self._terminal is not None:
            raise DuplicateTerminalError()
        return self._append(event, node, status, payload)

    def emit_terminal(
        self,
        outcome: TerminalOutcome,
        *,
        node: TurnNode = TurnNode.FINALIZE,
    ) -> WorkflowEvent:
        if self._terminal is not None:
            raise DuplicateTerminalError()
        self._terminal = outcome
        status = {
            "achieved": EventStatus.COMPLETED,
            "needs_clarification": EventStatus.COMPLETED,
            "insufficient_evidence": EventStatus.COMPLETED,
            "safe_failure": EventStatus.FAILED,
            "budget_exhausted": EventStatus.FAILED,
            "cancelled": EventStatus.CANCELLED,
            "system_failure": EventStatus.FAILED,
        }[outcome.status.value]
        return self._append(
            EventType.TURN_TERMINAL,
            node,
            status,
            {
                "terminal_status": outcome.status.value,
                "final_text": outcome.final_text,
                "goal_status": outcome.goal_status.value,
                "unmet_criteria": list(outcome.unmet_criteria),
                "evidence_ids": list(outcome.evidence_ids),
                "recoverable": outcome.recoverable,
                "route": outcome.route,
                "error_code": outcome.error_code,
                "metadata": dict(outcome.metadata),
            },
        )

    def _append(
        self,
        event: EventType,
        node: TurnNode,
        status: EventStatus,
        payload: Mapping[str, Any] | None,
    ) -> WorkflowEvent:
        item = WorkflowEvent(
            event=event,
            session_id=self.session_id,
            run_id=self.run_id,
            goal_id=self.goal_id,
            sequence=len(self._events),
            surface=self.surface,
            timestamp=datetime.now(UTC).isoformat(),
            node=node,
            status=status,
            payload=dict(payload or {}),
        )
        self._events.append(item)
        return item

    def __iter__(self) -> Iterator[WorkflowEvent]:
        return iter(tuple(self._events))
