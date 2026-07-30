from __future__ import annotations

from collections.abc import Iterable, Iterator
from uuid import uuid4

from soca.core.turn import RuntimeResult, RuntimeRoute, RuntimeStreamEvent

from .contracts import GoalStatus, TerminalOutcome, TerminalStatus, TurnNode
from .errors import DuplicateTerminalError
from .events import EventStatus, EventType, WorkflowEvent, WorkflowEventStream


def terminal_from_runtime_result(result: RuntimeResult) -> TerminalOutcome:
    evidence_ids: tuple[str, ...] = ()
    unmet_criteria: tuple[str, ...] = ()
    status = TerminalStatus.ACHIEVED
    goal_status = GoalStatus.ACHIEVED
    recoverable = False
    error_code: str | None = None

    if result.route is RuntimeRoute.CLARIFICATION:
        status = TerminalStatus.NEEDS_CLARIFICATION
        goal_status = GoalStatus.WAITING_FOR_USER
        recoverable = True
        unmet_criteria = ("clarification_required",)
    elif result.trace is not None and result.trace.evidence_status == "insufficient":
        status = TerminalStatus.INSUFFICIENT_EVIDENCE
        goal_status = GoalStatus.FAILED
        unmet_criteria = ("grounded_answer_or_abstention",)
    elif result.blocked:
        status = TerminalStatus.SAFE_FAILURE
        goal_status = GoalStatus.FAILED
        error_code = "runtime_blocked"

    if result.trace is not None:
        evidence_ids = tuple(
            str(getattr(item, "evidence_id", "") or getattr(item, "id", ""))
            for item in result.trace.evidence_decisions
            if getattr(item, "evidence_id", "") or getattr(item, "id", "")
        )

    return TerminalOutcome(
        status=status,
        final_text=result.response_text,
        goal_status=goal_status,
        unmet_criteria=unmet_criteria,
        evidence_ids=evidence_ids,
        recoverable=recoverable,
        route=result.route.value,
        error_code=error_code,
        metadata={"adapter": "runtime_result"},
    )


def iter_runtime_events(
    source: RuntimeResult | Iterable[RuntimeStreamEvent],
    *,
    turn_id: str = "",
    session_id: str = "legacy-session",
    surface: str = "chat",
) -> Iterator[WorkflowEvent]:
    run_id = turn_id.strip() or uuid4().hex
    goal_id = f"legacy-{run_id}"
    normalized_surface = surface if surface in {"ask", "cli", "chat", "voice"} else "chat"
    stream = WorkflowEventStream(
        session_id=session_id,
        run_id=run_id,
        goal_id=goal_id,
        surface=normalized_surface,  # type: ignore[arg-type]
    )
    yield stream.emit(
        EventType.TURN_STARTED,
        TurnNode.ADMIT,
        status=EventStatus.STARTED,
    )

    if isinstance(source, RuntimeResult):
        yield stream.emit(
            EventType.STEP_COMPLETED,
            TurnNode.SYNTHESIZE,
            status=EventStatus.COMPLETED,
            payload={"route": source.route.value},
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
            if event.type == "token":
                yield stream.emit(
                    EventType.ANSWER_DELTA,
                    TurnNode.SYNTHESIZE,
                    payload={"text": event.text},
                )
            else:
                yield stream.emit(
                    EventType.STEP_PROGRESS,
                    TurnNode.SYNTHESIZE,
                    payload={"stream_event": event.type, "text": event.text},
                )
    except Exception as exc:  # noqa: BLE001 - adapter must close the event contract
        if isinstance(exc, DuplicateTerminalError):
            raise
        if stream.terminal_outcome is None:
            yield stream.emit_terminal(
                TerminalOutcome(
                    status=TerminalStatus.SYSTEM_FAILURE,
                    final_text="",
                    goal_status=GoalStatus.FAILED,
                    error_code="stream_error",
                    metadata={"exception_type": type(exc).__name__},
                )
            )
        return

    if not saw_result:
        yield stream.emit_terminal(
            TerminalOutcome(
                status=TerminalStatus.SYSTEM_FAILURE,
                final_text="",
                goal_status=GoalStatus.FAILED,
                error_code="missing_terminal_result",
            )
        )
