from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from .contracts import TerminalOutcome, TurnState
from .errors import DuplicateTerminalError

EventKind = Literal["started", "node", "update", "terminal", "error"]


@dataclass(frozen=True)
class WorkflowEvent:
    sequence: int
    kind: EventKind
    state: TurnState
    payload: Mapping[str, Any] = field(default_factory=dict)
    terminal: bool = False


class WorkflowEventStream:
    """Ordered event collector with an exactly-once terminal guard."""

    def __init__(self, turn_id: str = "") -> None:
        self.turn_id = turn_id
        self._events: list[WorkflowEvent] = []
        self._terminal: TerminalOutcome | None = None

    @property
    def terminal_outcome(self) -> TerminalOutcome | None:
        return self._terminal

    def emit(
        self,
        kind: EventKind,
        state: TurnState,
        payload: Mapping[str, Any] | None = None,
    ) -> WorkflowEvent:
        if kind == "terminal":
            raise ValueError("use emit_terminal for terminal events")
        if self._terminal is not None:
            raise DuplicateTerminalError()
        event = WorkflowEvent(
            sequence=len(self._events),
            kind=kind,
            state=state,
            payload=dict(payload or {}),
        )
        self._events.append(event)
        return event

    def emit_terminal(self, outcome: TerminalOutcome) -> WorkflowEvent:
        if self._terminal is not None:
            raise DuplicateTerminalError()
        self._terminal = outcome
        state = {
            "succeeded": TurnState.COMPLETED,
            "failed": TurnState.FAILED,
            "cancelled": TurnState.CANCELLED,
        }[outcome.status.value]
        event = WorkflowEvent(
            sequence=len(self._events),
            kind="terminal",
            state=state,
            payload={
                "status": outcome.status.value,
                "response_text": outcome.response_text,
                "route": outcome.route,
                "error_code": outcome.error_code,
                "metadata": dict(outcome.metadata),
            },
            terminal=True,
        )
        self._events.append(event)
        return event

    def __iter__(self) -> Iterator[WorkflowEvent]:
        return iter(tuple(self._events))
