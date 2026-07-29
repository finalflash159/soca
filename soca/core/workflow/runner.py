from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from soca.tools import SideEffectLevel, ToolCall, ToolResult, ToolRuntime

from .budget import BudgetLedger, BudgetSnapshot
from .contracts import (
    GoalContract,
    TerminalOutcome,
    TerminalStatus,
    TurnBudget,
    TurnState,
)
from .errors import BudgetExceededError, WorkflowError, WorkflowErrorCode
from .events import WorkflowEvent, WorkflowEventStream
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
    revision = goal.metadata.get("revision", 0)
    canonical = json.dumps(
        {
            "goal_id": goal.goal_id,
            "revision": revision,
            "tool": call.name,
            "arguments": call.arguments,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class InvalidTransitionError(WorkflowError):
    def __init__(self, source: TurnState, target: TurnState) -> None:
        super().__init__(
            code=WorkflowErrorCode.PROTOCOL_ERROR,
            message=f"invalid workflow transition: {source.value} -> {target.value}",
        )
        self.source = source
        self.target = target


_TRANSITIONS: Mapping[TurnState, frozenset[TurnState]] = {
    TurnState.RECEIVED: frozenset({TurnState.ANALYZING, TurnState.CANCELLED}),
    TurnState.ANALYZING: frozenset({TurnState.PLANNING, TurnState.AUTHORIZING, TurnState.CANCELLED}),
    TurnState.PLANNING: frozenset({TurnState.AUTHORIZING, TurnState.SYNTHESIZING, TurnState.CANCELLED}),
    TurnState.AUTHORIZING: frozenset({TurnState.EXECUTING, TurnState.FAILED, TurnState.CANCELLED}),
    TurnState.EXECUTING: frozenset({TurnState.OBSERVING, TurnState.FAILED, TurnState.CANCELLED}),
    TurnState.OBSERVING: frozenset({TurnState.VERIFYING, TurnState.EXECUTING, TurnState.FAILED, TurnState.CANCELLED}),
    TurnState.VERIFYING: frozenset({TurnState.AUTHORIZING, TurnState.SYNTHESIZING, TurnState.FAILED, TurnState.CANCELLED}),
    TurnState.SYNTHESIZING: frozenset({TurnState.COMPLETED, TurnState.FAILED, TurnState.CANCELLED}),
}


@dataclass
class ControlledWorkflowRunner:
    """Run a bounded, read-mostly workflow without changing legacy runtime."""

    tool_runtime: ToolRuntime
    verifier: DeterministicVerifier = field(default_factory=DeterministicVerifier)
    budget: TurnBudget = field(default_factory=TurnBudget)

    def run(
        self,
        goal: GoalContract,
        *,
        planner: WorkflowPlanner | None = None,
        explicit_call: ToolCall | None = None,
        authorize: AuthorizationPolicy | None = None,
        cancelled: CancellationCheck | None = None,
        turn_id: str = "",
    ) -> WorkflowRun:
        ledger = BudgetLedger(self.budget)
        retries = RetryLedger()
        stream = WorkflowEventStream(turn_id=turn_id)
        observations: list[ToolResult] = []
        current = TurnState.RECEIVED
        plan: ActionPlan | None = None

        def is_cancelled() -> bool:
            return bool(cancelled and cancelled())

        def transition(target: TurnState) -> None:
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

        stream.emit("started", TurnState.RECEIVED, {"turn_id": turn_id, "goal_id": goal.goal_id})
        try:
            if is_cancelled():
                return terminal(self._cancelled_outcome("cancelled_before_start"))
            transition(TurnState.ANALYZING)
            stream.emit(
                "update",
                current,
                {"phase": "analyzing", "text": "Đang phân tích mục tiêu"},
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
                transition(TurnState.PLANNING)
                stream.emit(
                    "update",
                    current,
                    {"phase": "planning", "text": "Đang lập kế hoạch"},
                )
                self._bind_model_budget(planner, ledger)
                if not self._has_model_budget_hook(planner):
                    ledger.consume("model")
                plan = planner.plan(goal.statement)

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
            if current != TurnState.SYNTHESIZING:
                transition(TurnState.SYNTHESIZING)
            stream.emit(
                "update",
                current,
                {"phase": "synthesizing", "text": "Đang tổng hợp kết quả"},
            )
            return terminal(
                TerminalOutcome(
                    status=TerminalStatus.SUCCEEDED,
                    response_text=plan.final_instruction,
                    route="controlled_workflow",
                    metadata={
                        "goal_id": goal.goal_id,
                        "observations": len(observations),
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
        current: TurnState,
        authorize: AuthorizationPolicy | None,
        is_cancelled: Callable[[], bool],
    ) -> tuple[TurnState, ToolResult, Verification, TerminalOutcome | None]:
        fingerprint = action_fingerprint(goal, step.call)
        tool = self.tool_runtime.get(step.call.name)
        if tool is None or not tool.spec.enabled:
            return current, ToolResult(step.call.name, False, "", error="unknown_tool"), Verification(False, "unknown_tool"), self._failed_outcome("unknown_tool")
        needs_authorization = step.requires_authorization or tool.spec.side_effect != SideEffectLevel.READ_ONLY
        if needs_authorization:
            if authorize is None or not authorize(goal, step):
                return current, ToolResult(step.call.name, False, "", error="authorization_denied"), Verification(False, "authorization_denied"), self._failed_outcome("authorization_denied")

        if retries.is_completed(fingerprint):
            result = ToolResult(step.call.name, False, "", error="duplicate_action")
            return current, result, Verification(False, "duplicate_action"), self._failed_outcome("duplicate_action")

        attempt = retries.attempts(fingerprint)
        while True:
            if is_cancelled():
                result = ToolResult(step.call.name, False, "", error="cancelled")
                return current, result, Verification(False, "cancelled"), self._cancelled_outcome("cancelled_during_action")
            if current != TurnState.AUTHORIZING:
                current = self._move(current, TurnState.AUTHORIZING, ledger, stream)
            stream.emit(
                "update",
                current,
                {"phase": "authorizing", "action_id": step.action_id, "text": "Đang chuẩn bị công cụ"},
            )
            current = self._move(current, TurnState.EXECUTING, ledger, stream)
            ledger.consume("tool")
            retries.record_attempt(fingerprint)
            stream.emit(
                "update",
                current,
                {"phase": "executing", "action_id": step.action_id, "text": "Đang thực thi"},
            )
            result = ToolExecutionNode(self.tool_runtime).execute(step.call).output
            if not isinstance(result, ToolResult):
                result = ToolResult(step.call.name, False, "", error="invalid_tool_result")
            if result.ok:
                retries.mark_completed(fingerprint)
            current = self._move(current, TurnState.OBSERVING, ledger, stream)
            stream.emit(
                "update",
                current,
                {"phase": "observing", "action_id": step.action_id, "text": "Đã nhận kết quả"},
            )
            verification = self.verifier.verify(goal, result)
            current = self._move(current, TurnState.VERIFYING, ledger, stream)
            stream.emit(
                "update",
                current,
                {"phase": "verifying", "action_id": step.action_id, "achieved": verification.achieved},
            )
            if verification.achieved:
                return current, result, verification, None
            if not result.ok and attempt < self.budget.max_retries:
                ledger.consume("retry")
                attempt += 1
                stream.emit(
                    "update",
                    current,
                    {"phase": "retrying", "action_id": step.action_id, "text": "Đang thử lại"},
                )
                continue
            return current, result, verification, None

    @staticmethod
    def _move(
        current: TurnState,
        target: TurnState,
        ledger: BudgetLedger,
        stream: WorkflowEventStream,
    ) -> TurnState:
        if target not in _TRANSITIONS.get(current, frozenset()):
            raise InvalidTransitionError(current, target)
        ledger.consume("transition")
        stream.emit("node", target, {"node": target.value})
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
        return TerminalOutcome(
            status=TerminalStatus.FAILED,
            route="controlled_workflow",
            error_code=code,
            metadata=metadata,
        )

    @staticmethod
    def _cancelled_outcome(code: str) -> TerminalOutcome:
        return TerminalOutcome(
            status=TerminalStatus.CANCELLED,
            route="controlled_workflow",
            error_code=code,
        )
