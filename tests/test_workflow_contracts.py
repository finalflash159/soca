from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from soca.core.turn import RuntimeResult, RuntimeRoute, RuntimeStreamEvent, iter_workflow_events
from soca.core.workflow import (
    Advance,
    BudgetExceededError,
    BudgetLedger,
    Capability,
    DuplicateTerminalError,
    EventStatus,
    EventType,
    GoalConstraint,
    GoalContract,
    GoalStatus,
    SourceKind,
    SuccessCriterion,
    TerminalOutcome,
    TerminalStatus,
    TurnBudget,
    TurnNode,
    WorkflowEventStream,
)
from soca.core.workflow.protocol import workflow_event_from_protocol, workflow_event_to_protocol


def goal() -> GoalContract:
    return GoalContract(
        goal_id="goal-1",
        objective="  kiểm tra ghi chú  ",
        success_criteria=(SuccessCriterion("source_queried", source=SourceKind.KNOWLEDGE),),
        constraints=(GoalConstraint("language", "vi"),),
        required_sources=(SourceKind.KNOWLEDGE,),
    )


def outcome(status: TerminalStatus = TerminalStatus.ACHIEVED) -> TerminalOutcome:
    return TerminalOutcome(
        status=status,
        final_text="ok" if status is TerminalStatus.ACHIEVED else "",
        goal_status=(
            GoalStatus.ACHIEVED if status is TerminalStatus.ACHIEVED else GoalStatus.FAILED
        ),
    )


def stream() -> WorkflowEventStream:
    return WorkflowEventStream(
        session_id="session-1",
        run_id="run-1",
        goal_id="goal-1",
        surface="chat",
    )


def test_goal_contract_normalizes_objective_and_is_immutable() -> None:
    contract = goal()

    assert contract.objective == "kiểm tra ghi chú"
    assert contract.required_sources == (SourceKind.KNOWLEDGE,)
    with pytest.raises(FrozenInstanceError):
        contract.objective = "khác"  # type: ignore[misc]


def test_nested_action_arguments_are_immutable() -> None:
    from soca.core.workflow import PlannedAction

    action = PlannedAction(
        action_id="a1",
        capability=Capability.KNOWLEDGE_SEARCH,
        tool_name="knowledge.search",
        arguments={"filters": {"tags": ["ml", "notes"]}},
        purpose="find evidence",
        expected_observation="ranked notes",
        required=True,
    )

    filters = action.arguments["filters"]
    assert isinstance(filters, dict) is False
    with pytest.raises(TypeError):
        filters["tags"] = ()  # type: ignore[index]


def test_node_outcome_is_a_typed_sum_member() -> None:
    advance = Advance(next_node=TurnNode.RESOLVE_GOAL)

    assert advance.next_node is TurnNode.RESOLVE_GOAL


def test_budget_ledger_enforces_each_counter() -> None:
    ledger = BudgetLedger(TurnBudget(max_transitions=1, max_tool_calls=1))

    ledger.consume("transition")
    ledger.consume("tool")
    with pytest.raises(BudgetExceededError):
        ledger.consume("transition")


def test_event_stream_allows_exactly_one_terminal_event() -> None:
    events = stream()
    started = events.emit(
        EventType.TURN_STARTED,
        TurnNode.ADMIT,
        status=EventStatus.STARTED,
    )
    terminal = events.emit_terminal(outcome())

    assert [event.sequence for event in events] == [0, 1]
    assert started.terminal is False
    assert terminal.terminal is True
    with pytest.raises(ValueError):
        events.emit(EventType.TURN_TERMINAL, TurnNode.FINALIZE)
    with pytest.raises(DuplicateTerminalError):
        events.emit_terminal(outcome(TerminalStatus.SYSTEM_FAILURE))


def test_event_stream_rejects_an_invalid_surface_without_coercion() -> None:
    with pytest.raises(ValueError, match="surface"):
        WorkflowEventStream(
            session_id="session-1",
            run_id="run-1",
            goal_id="goal-1",
            surface="desktop",  # type: ignore[arg-type]
        )


def test_blocking_and_streaming_legacy_results_share_terminal_shape() -> None:
    result = RuntimeResult(response_text="ok", route=RuntimeRoute.FREE_CHAT)
    blocking = list(iter_workflow_events(result, turn_id="blocking"))
    streaming = list(
        iter_workflow_events(
            iter(
                [
                    RuntimeStreamEvent(type="token", text="ok"),
                    RuntimeStreamEvent(type="result", result=result),
                ]
            ),
            turn_id="streaming",
        )
    )

    assert blocking[-1].event is EventType.TURN_TERMINAL
    assert streaming[-1].event is EventType.TURN_TERMINAL
    assert (
        blocking[-1].payload["terminal_status"]
        == streaming[-1].payload["terminal_status"]
        == "achieved"
    )


def test_stream_without_result_emits_system_failure_terminal() -> None:
    events = list(iter_workflow_events(iter([RuntimeStreamEvent(type="token", text="partial")])))

    assert events[-1].event is EventType.TURN_TERMINAL
    assert events[-1].payload["terminal_status"] == "system_failure"
    assert events[-1].payload["error_code"] == "missing_terminal_result"


def test_stream_exception_emits_system_failure_terminal() -> None:
    def failing_source():
        yield RuntimeStreamEvent(type="token", text="partial")
        raise RuntimeError("boom")

    events = list(iter_workflow_events(failing_source()))

    assert events[-1].payload["terminal_status"] == "system_failure"
    assert events[-1].payload["error_code"] == "stream_error"
    assert events[-1].payload["metadata"] == {"exception_type": "RuntimeError"}


def test_terminal_metadata_cannot_overwrite_canonical_fields() -> None:
    events = stream()
    terminal = events.emit_terminal(
        TerminalOutcome(
            status=TerminalStatus.ACHIEVED,
            final_text="ok",
            goal_status=GoalStatus.ACHIEVED,
            route="free_chat",
            metadata={
                "terminal_status": "system_failure",
                "final_text": "wrong",
                "route": "wrong",
            },
        )
    )

    assert terminal.status is EventStatus.COMPLETED
    assert terminal.payload["terminal_status"] == "achieved"
    assert terminal.payload["final_text"] == "ok"
    assert terminal.payload["route"] == "free_chat"
    assert terminal.payload["metadata"]["terminal_status"] == "system_failure"


def test_protocol_event_contains_complete_v2_envelope() -> None:
    event = stream().emit(
        EventType.GOAL_RESOLVED,
        TurnNode.RESOLVE_GOAL,
        status=EventStatus.COMPLETED,
    )

    payload = workflow_event_to_protocol(event)

    assert payload["protocol_version"] == 3
    assert payload["event"] == "goal_resolved"
    assert payload["session_id"] == "session-1"
    assert payload["run_id"] == "run-1"
    assert payload["goal_id"] == "goal-1"
    assert payload["sequence"] == 0
    assert payload["surface"] == "chat"
    assert payload["node"] == "resolve_goal"
    assert payload["status"] == "completed"
    assert payload["timestamp"]
    assert workflow_event_from_protocol(payload) == event


def test_protocol_rejects_an_incomplete_workflow_envelope() -> None:
    payload = workflow_event_to_protocol(stream().emit(EventType.TURN_STARTED, TurnNode.ADMIT))
    del payload["goal_id"]

    with pytest.raises(ValueError, match="invalid string field"):
        workflow_event_from_protocol(payload)


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
