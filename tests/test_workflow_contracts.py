from __future__ import annotations

import pytest

from soca.core.turn import RuntimeResult, RuntimeRoute, RuntimeStreamEvent, iter_workflow_events
from soca.core.workflow import (
    BudgetExceededError,
    BudgetLedger,
    DuplicateTerminalError,
    GoalContract,
    TerminalOutcome,
    TerminalStatus,
    TurnBudget,
    TurnState,
    WorkflowEventStream,
)


def test_goal_contract_normalizes_statement_and_freezes_metadata() -> None:
    goal = GoalContract(statement="  kiểm tra ghi chú  ", metadata={"source": "voice"})

    assert goal.statement == "kiểm tra ghi chú"
    assert goal.metadata["source"] == "voice"
    with pytest.raises(TypeError):
        goal.metadata["source"] = "text"  # type: ignore[index]


def test_budget_ledger_enforces_each_counter() -> None:
    ledger = BudgetLedger(TurnBudget(max_transitions=1, max_tool_calls=1))

    ledger.consume("transition")
    ledger.consume("tool")
    with pytest.raises(BudgetExceededError):
        ledger.consume("transition")


def test_event_stream_allows_exactly_one_terminal_event() -> None:
    stream = WorkflowEventStream(turn_id="t1")
    started = stream.emit("started", TurnState.RECEIVED)
    terminal = stream.emit_terminal(TerminalOutcome(status=TerminalStatus.SUCCEEDED, response_text="ok"))

    assert [event.sequence for event in stream] == [0, 1]
    assert started.terminal is False
    assert terminal.terminal is True
    with pytest.raises(ValueError):
        stream.emit("terminal", TurnState.COMPLETED)
    with pytest.raises(DuplicateTerminalError):
        stream.emit_terminal(TerminalOutcome(status=TerminalStatus.FAILED))


def test_blocking_and_streaming_legacy_results_share_terminal_shape() -> None:
    result = RuntimeResult(response_text="ok", route=RuntimeRoute.FREE_CHAT)
    blocking = list(iter_workflow_events(result))
    streaming = list(
        iter_workflow_events(
            iter(
                [
                    RuntimeStreamEvent(type="token", text="ok"),
                    RuntimeStreamEvent(type="result", result=result),
                ]
            )
        )
    )

    assert blocking[-1].kind == "terminal"
    assert streaming[-1].kind == "terminal"
    assert blocking[-1].payload["status"] == streaming[-1].payload["status"] == "succeeded"


def test_stream_without_result_emits_failed_terminal_not_complete() -> None:
    events = list(iter_workflow_events(iter([RuntimeStreamEvent(type="token", text="partial")])))

    assert events[-1].kind == "terminal"
    assert events[-1].payload["status"] == "failed"
    assert events[-1].payload["error_code"] == "missing_terminal_result"


def test_duplicate_stream_results_are_rejected_instead_of_emitting_two_terminals() -> None:
    result = RuntimeResult(response_text="ok", route=RuntimeRoute.FREE_CHAT)
    source = iter(
        [
            RuntimeStreamEvent(type="result", result=result),
            RuntimeStreamEvent(type="result", result=result),
        ]
    )

    with pytest.raises(DuplicateTerminalError):
        list(iter_workflow_events(source))
