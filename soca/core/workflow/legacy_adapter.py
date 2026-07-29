from __future__ import annotations

from collections.abc import Iterable, Iterator

from soca.core.turn import RuntimeResult, RuntimeStreamEvent

from .contracts import TerminalOutcome, TerminalStatus, TurnState
from .events import WorkflowEvent, WorkflowEventStream
from .errors import DuplicateTerminalError


def terminal_from_runtime_result(result: RuntimeResult) -> TerminalOutcome:
    return TerminalOutcome(
        status=TerminalStatus.FAILED if result.blocked else TerminalStatus.SUCCEEDED,
        response_text=result.response_text,
        route=result.route.value,
        error_code="runtime_blocked" if result.blocked else None,
        metadata={"legacy_route": result.route.value},
    )


def iter_runtime_events(
    source: RuntimeResult | Iterable[RuntimeStreamEvent],
    *,
    turn_id: str = "",
) -> Iterator[WorkflowEvent]:
    """Adapt blocking and streaming legacy runtime results to one event shape."""
    stream = WorkflowEventStream(turn_id=turn_id)
    yield stream.emit("started", TurnState.RECEIVED, {"turn_id": turn_id})

    if isinstance(source, RuntimeResult):
        yield stream.emit(
            "node",
            TurnState.SYNTHESIZING,
            {"response_text": source.response_text, "route": source.route.value},
        )
        yield stream.emit_terminal(terminal_from_runtime_result(source))
        return

    saw_result = False
    try:
        for event in source:
            if event.type == "result" and event.result is not None:
                saw_result = True
                yield stream.emit_terminal(terminal_from_runtime_result(event.result))
                continue
            yield stream.emit(
                "update",
                TurnState.SYNTHESIZING,
                {"type": event.type, "text": event.text},
            )
    except Exception as exc:  # noqa: BLE001 - adapter must close the event contract
        if isinstance(exc, DuplicateTerminalError):
            raise
        if stream.terminal_outcome is None:
            yield stream.emit_terminal(
                TerminalOutcome(
                    status=TerminalStatus.FAILED,
                    error_code="stream_error",
                    metadata={"exception_type": type(exc).__name__},
                )
            )
        return

    if not saw_result:
        yield stream.emit_terminal(
            TerminalOutcome(
                status=TerminalStatus.FAILED,
                error_code="missing_terminal_result",
            )
        )
