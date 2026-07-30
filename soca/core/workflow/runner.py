from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import uuid4

from soca.core.guardrails import (
    DEFAULT_POLICY,
    GuardrailPolicy,
    check_final_output,
    check_tool_call,
)
from soca.tools import SideEffectLevel, ToolCall, ToolResult, ToolRuntime

from .budget import BudgetLedger, BudgetSnapshot
from .contracts import (
    GoalContract,
    GoalStatus,
    TerminalOutcome,
    TerminalStatus,
    TurnBudget,
    TurnNode,
    TurnSource,
)
from .errors import BudgetExceededError, WorkflowError, WorkflowErrorCode
from .events import EventStatus, EventType, WorkflowEvent, WorkflowEventStream
from .nodes import ToolExecutionNode
from .planner import ActionPlan, PlanStep, WorkflowPlanner
from .verifier import DeterministicVerifier, Verification


class AuthorizationPolicy(Protocol):
    def __call__(self, goal: GoalContract, step: PlanStep) -> bool:
        ...


class CancellationCheck(Protocol):
    def __call__(self) -> bool:
        ...


@dataclass(frozen=True)
class WorkflowRun:
    events: tuple[WorkflowEvent, ...]
    terminal: TerminalOutcome
    budget: BudgetSnapshot
    observations: tuple[ToolResult, ...] = ()


class RetryLedger:
    """Track attempts once per canonical action fingerprint for one run."""

    def __init__(self) -> None:
        self._attempts: dict[str, int] = {}
        self._completed: set[str] = set()

    def attempts(self, fingerprint: str) -> int:
        return self._attempts.get(fingerprint, 0)

    def record_attempt(self, fingerprint: str) -> int:
        count = self._attempts.get(fingerprint, 0) + 1
        self._attempts[fingerprint] = count
        return count

    def mark_completed(self, fingerprint: str) -> None:
        self._completed.add(fingerprint)

    def is_completed(self, fingerprint: str) -> bool:
        return fingerprint in self._completed


def action_fingerprint(goal: GoalContract, call: ToolCall) -> str:
    """Create a stable idempotency key without storing user content in the key."""
    canonical = json.dumps(
        {
            "goal_id": goal.goal_id,
            "goal_revision": {
                "objective": goal.objective,
                "constraints": [
                    {"kind": item.kind, "value": _jsonable(item.value)}
                    for item in goal.constraints
                ],
                "required_sources": [item.value for item in goal.required_sources],
                "resolved_entities": [
                    {
                        "surface": item.surface,
                        "canonical": item.canonical,
                        "confidence": item.confidence,
                    }
                    for item in goal.resolved_entities
                ],
            },
            "tool": call.name,
            "arguments": call.arguments,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class InvalidTransitionError(WorkflowError):
    def __init__(self, source: TurnNode, target: TurnNode) -> None:
        super().__init__(
            code=WorkflowErrorCode.PROTOCOL_ERROR,
            message=f"invalid workflow transition: {source.value} -> {target.value}",
        )
        self.source = source
        self.target = target


_TRANSITIONS: Mapping[TurnNode, frozenset[TurnNode]] = {
    TurnNode.ADMIT: frozenset({TurnNode.RESOLVE_GOAL}),
    TurnNode.RESOLVE_GOAL: frozenset({TurnNode.MAKE_PLAN, TurnNode.AUTHORIZE_ACTION}),
    TurnNode.MAKE_PLAN: frozenset({TurnNode.AUTHORIZE_ACTION, TurnNode.SYNTHESIZE}),
    TurnNode.AUTHORIZE_ACTION: frozenset({TurnNode.EXECUTE_ACTION}),
    TurnNode.EXECUTE_ACTION: frozenset({TurnNode.ASSESS_OBSERVATION}),
    TurnNode.ASSESS_OBSERVATION: frozenset(
        {TurnNode.VERIFY_ANSWER, TurnNode.EXECUTE_ACTION}
    ),
    TurnNode.VERIFY_ANSWER: frozenset(
        {TurnNode.AUTHORIZE_ACTION, TurnNode.SYNTHESIZE}
    ),
    TurnNode.SYNTHESIZE: frozenset({TurnNode.FINALIZE}),
}


@dataclass
class ControlledWorkflowRunner:
    """Run a bounded, read-mostly workflow without changing legacy runtime."""

    tool_runtime: ToolRuntime
    verifier: DeterministicVerifier = field(default_factory=DeterministicVerifier)
    budget: TurnBudget = field(default_factory=TurnBudget)
    guardrail_policy: GuardrailPolicy = DEFAULT_POLICY

    def run(
        self,
        goal: GoalContract,
        *,
        planner: WorkflowPlanner | None = None,
        explicit_call: ToolCall | None = None,
        authorize: AuthorizationPolicy | None = None,
        cancelled: CancellationCheck | None = None,
        turn_id: str = "",
        session_id: str = "in-memory-session",
        surface: TurnSource = "chat",
    ) -> WorkflowRun:
        run_id = turn_id.strip() or f"run-{uuid4().hex}"
        ledger = BudgetLedger(self.budget)
        retries = RetryLedger()
        stream = WorkflowEventStream(
            session_id=session_id,
            run_id=run_id,
            goal_id=goal.goal_id,
            surface=surface,
        )
        observations: list[ToolResult] = []
        current = TurnNode.ADMIT
        plan: ActionPlan | None = None

        def is_cancelled() -> bool:
            return bool(cancelled and cancelled())

        def transition(target: TurnNode) -> None:
            nonlocal current
            current = self._move(current, target, ledger, stream)

        def terminal(outcome: TerminalOutcome) -> WorkflowRun:
            if stream.terminal_outcome is None:
                stream.emit_terminal(outcome)
            return WorkflowRun(
                events=tuple(stream),
                terminal=outcome,
                budget=ledger.snapshot(),
                observations=tuple(observations),
            )

        stream.emit(
            EventType.TURN_STARTED,
            TurnNode.ADMIT,
            status=EventStatus.STARTED,
        )
        try:
            if is_cancelled():
                return terminal(self._cancelled_outcome("cancelled_before_start"))
            transition(TurnNode.RESOLVE_GOAL)
            stream.emit(
                EventType.GOAL_RESOLVED,
                current,
                status=EventStatus.COMPLETED,
                payload={"objective": goal.objective},
            )

            if explicit_call is not None:
                plan = ActionPlan(
                    steps=(
                        PlanStep(
                            action_id="explicit-1",
                            call=explicit_call,
                            purpose="explicit_command",
                        ),
                    ),
                    final_instruction="",
                    rationale="explicit_command",
                )
            else:
                if planner is None:
                    return terminal(self._failed_outcome("planner_required"))
                transition(TurnNode.MAKE_PLAN)
                stream.emit(
                    EventType.STEP_PROGRESS,
                    current,
                    payload={"operation": "plan"},
                )
                self._bind_model_budget(planner, ledger)
                if not self._has_model_budget_hook(planner):
                    ledger.consume("model")
                plan = planner.plan(goal.planner_text())

            if not plan.steps:
                return terminal(self._failed_outcome("empty_plan"))

            for step in plan.steps:
                if is_cancelled():
                    return terminal(self._cancelled_outcome("cancelled_before_action"))
                result = self._run_step(
                    goal,
                    step,
                    ledger=ledger,
                    retries=retries,
                    stream=stream,
                    current=current,
                    authorize=authorize,
                    is_cancelled=is_cancelled,
                )
                current = result[0]
                observations.append(result[1])
                verification = result[2]
                if result[3] is not None:
                    return terminal(result[3])
                if not verification.achieved:
                    return terminal(self._failed_outcome(verification.reason))

            if is_cancelled():
                return terminal(self._cancelled_outcome("cancelled_before_answer"))
            if current != TurnNode.SYNTHESIZE:
                transition(TurnNode.SYNTHESIZE)
            stream.emit(
                EventType.STEP_STARTED,
                current,
                status=EventStatus.STARTED,
                payload={"operation": "synthesize"},
            )
            response_text = self._observation_text(observations)
            output_event = check_final_output(
                response_text,
                knowledge_used=any(
                    result.name in {"knowledge.search", "knowledge.read", "memory.search"}
                    for result in observations
                ),
                citations=self._citation_payload(observations),
                tool_results=tuple(observations),
                policy=self.guardrail_policy,
            )
            if output_event.blocked:
                return terminal(self._failed_outcome("output_guardrail", detail=output_event.reason))
            transition(TurnNode.FINALIZE)
            return terminal(
                TerminalOutcome(
                    status=TerminalStatus.ACHIEVED,
                    final_text=response_text,
                    goal_status=GoalStatus.ACHIEVED,
                    route="controlled_workflow",
                    metadata={
                        "goal_id": goal.goal_id,
                        "observations": len(observations),
                        "planner_instruction": plan.final_instruction,
                        "rationale": plan.rationale,
                    },
                )
            )
        except BudgetExceededError as exc:
            return terminal(self._failed_outcome("budget_exhausted", detail=exc.budget_name))
        except InvalidTransitionError as exc:
            return terminal(self._failed_outcome("invalid_transition", detail=str(exc)))
        except Exception as exc:  # noqa: BLE001 - controller boundary must terminalize
            return terminal(
                self._failed_outcome(
                    "workflow_error",
                    detail=type(exc).__name__,
                )
            )

    def _run_step(
        self,
        goal: GoalContract,
        step: PlanStep,
        *,
        ledger: BudgetLedger,
        retries: RetryLedger,
        stream: WorkflowEventStream,
        current: TurnNode,
        authorize: AuthorizationPolicy | None,
        is_cancelled: Callable[[], bool],
    ) -> tuple[TurnNode, ToolResult, Verification, TerminalOutcome | None]:
        fingerprint = action_fingerprint(goal, step.call)
        tool = self.tool_runtime.get(step.call.name)
        if tool is None or not tool.spec.enabled:
            return (
                current,
                ToolResult(step.call.name, False, "", error="unknown_tool"),
                Verification(False, "unknown_tool"),
                self._failed_outcome("unknown_tool"),
            )
        guardrail_event = check_tool_call(
            self.tool_runtime,
            step.call,
            self.guardrail_policy,
        )
        if guardrail_event.blocked:
            result = ToolResult(
                step.call.name,
                False,
                "",
                error=guardrail_event.reason,
            )
            return (
                current,
                result,
                Verification(False, guardrail_event.reason),
                self._failed_outcome("guardrail_blocked", detail=guardrail_event.reason),
            )
        needs_authorization = step.requires_authorization or tool.spec.side_effect != SideEffectLevel.READ_ONLY
        if needs_authorization:
            if authorize is None or not authorize(goal, step):
                return (
                    current,
                    ToolResult(step.call.name, False, "", error="authorization_denied"),
                    Verification(False, "authorization_denied"),
                    self._failed_outcome("authorization_denied"),
                )

        if retries.is_completed(fingerprint):
            result = ToolResult(step.call.name, False, "", error="duplicate_action")
            return (
                current,
                result,
                Verification(False, "duplicate_action"),
                self._failed_outcome("duplicate_action"),
            )

        while True:
            if is_cancelled():
                result = ToolResult(step.call.name, False, "", error="cancelled")
                return current, result, Verification(False, "cancelled"), self._cancelled_outcome("cancelled_during_action")
            if current != TurnNode.AUTHORIZE_ACTION:
                current = self._move(current, TurnNode.AUTHORIZE_ACTION, ledger, stream)
            stream.emit(
                EventType.STEP_STARTED,
                current,
                status=EventStatus.STARTED,
                payload={"action_id": step.action_id, "operation": "authorize"},
            )
            current = self._move(current, TurnNode.EXECUTE_ACTION, ledger, stream)
            ledger.consume("tool")
            retries.record_attempt(fingerprint)
            stream.emit(
                EventType.STEP_PROGRESS,
                current,
                payload={"action_id": step.action_id, "operation": "execute"},
            )
            result = ToolExecutionNode(self.tool_runtime).execute(step.call)
            if not isinstance(result, ToolResult):
                result = ToolResult(step.call.name, False, "", error="invalid_tool_result")
            if result.ok:
                retries.mark_completed(fingerprint)
            current = self._move(current, TurnNode.ASSESS_OBSERVATION, ledger, stream)
            stream.emit(
                EventType.STEP_COMPLETED,
                current,
                status=EventStatus.COMPLETED if result.ok else EventStatus.FAILED,
                payload={"action_id": step.action_id, "result_ok": result.ok},
            )
            verification = self.verifier.verify(goal, result)
            current = self._move(current, TurnNode.VERIFY_ANSWER, ledger, stream)
            stream.emit(
                EventType.VERIFICATION_COMPLETED,
                current,
                status=(
                    EventStatus.COMPLETED if verification.achieved else EventStatus.FAILED
                ),
                payload={"action_id": step.action_id, "passed": verification.achieved},
            )
            if verification.achieved:
                return current, result, verification, None
            if (
                not result.ok
                and ledger.snapshot().retries < self.budget.max_readonly_tool_retries
                and (tool.spec.side_effect == SideEffectLevel.READ_ONLY or tool.spec.idempotent)
            ):
                ledger.consume("retry")
                stream.emit(
                    EventType.STEP_PROGRESS,
                    current,
                    payload={"action_id": step.action_id, "operation": "retry"},
                )
                continue
            return current, result, verification, None

    @staticmethod
    def _observation_text(observations: list[ToolResult]) -> str:
        parts: list[str] = []
        for result in observations:
            if result.content.strip():
                parts.append(result.content.strip())
            elif result.data:
                parts.append(json.dumps(result.data, ensure_ascii=False, sort_keys=True))
        return "\n\n".join(parts).strip()

    @staticmethod
    def _citation_payload(observations: list[ToolResult]) -> tuple[object, ...]:
        citations: list[object] = []
        for result in observations:
            hits = result.data.get("hits")
            if isinstance(hits, list):
                citations.extend(hits)
            path = result.data.get("path")
            if isinstance(path, str) and path:
                citations.append({"path": path})
        return tuple(citations)

    @staticmethod
    def _move(
        current: TurnNode,
        target: TurnNode,
        ledger: BudgetLedger,
        stream: WorkflowEventStream,
    ) -> TurnNode:
        if target not in _TRANSITIONS.get(current, frozenset()):
            raise InvalidTransitionError(current, target)
        ledger.consume("transition")
        stream.emit(
            EventType.STEP_STARTED,
            target,
            status=EventStatus.STARTED,
            payload={"operation": target.value},
        )
        return target

    @staticmethod
    def _bind_model_budget(planner: WorkflowPlanner, ledger: BudgetLedger) -> None:
        hook = getattr(planner, "set_model_call_hook", None)
        if callable(hook):
            hook(lambda: ledger.consume("model"))

    @staticmethod
    def _has_model_budget_hook(planner: WorkflowPlanner) -> bool:
        return callable(getattr(planner, "set_model_call_hook", None))

    @staticmethod
    def _failed_outcome(code: str, *, detail: str = "") -> TerminalOutcome:
        metadata: dict[str, Any] = {}
        if detail:
            metadata["detail"] = detail
        if code == "budget_exhausted":
            terminal_status = TerminalStatus.BUDGET_EXHAUSTED
        elif code == "no_matching_observation":
            terminal_status = TerminalStatus.INSUFFICIENT_EVIDENCE
        elif code in {
            "authorization_denied",
            "duplicate_action",
            "guardrail_blocked",
            "output_guardrail",
            "unknown_tool",
        }:
            terminal_status = TerminalStatus.SAFE_FAILURE
        else:
            terminal_status = TerminalStatus.SYSTEM_FAILURE
        return TerminalOutcome(
            status=terminal_status,
            final_text="",
            goal_status=GoalStatus.FAILED,
            route="controlled_workflow",
            error_code=code,
            metadata=metadata,
        )

    @staticmethod
    def _cancelled_outcome(code: str) -> TerminalOutcome:
        return TerminalOutcome(
            status=TerminalStatus.CANCELLED,
            final_text="",
            goal_status=GoalStatus.ACTIVE,
            route="controlled_workflow",
            error_code=code,
        )
