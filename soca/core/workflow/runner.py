from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Protocol, cast
from uuid import uuid4

from soca.core.guardrails import (
    DEFAULT_POLICY,
    GuardrailPolicy,
    check_final_output,
    check_tool_call,
)
from soca.tools import (
    SideEffectLevel,
    ToolCall,
    ToolExecutionStatus,
    ToolResult,
    ToolRuntime,
)

from .budget import BudgetLedger, BudgetSnapshot
from .contracts import (
    Advance,
    Capability,
    GoalContract,
    GoalStatus,
    NodeTrace,
    Observation,
    SourceKind,
    StatePatch,
    TerminalOutcome,
    TerminalStatus,
    TurnBudget,
    TurnNode,
    TurnSource,
    TurnState,
    VerificationReport,
)
from .errors import (
    BudgetExceededError,
    WorkflowError,
    WorkflowErrorCode,
    WorkflowRevisionError,
    WorkflowSynthesisError,
)
from .events import EventStatus, EventType, WorkflowEvent, WorkflowEventStream
from .nodes import ToolExecutionNode
from .planner import ActionPlan, PlanOutputError, PlanStep, WorkflowPlanner
from .verifier import (
    DeterministicVerifier,
    Verification,
    source_for_capability,
    tool_error_code,
    unmet_goal_criteria,
)


class AuthorizationPolicy(Protocol):
    def __call__(self, goal: GoalContract, step: PlanStep) -> bool: ...


class CancellationCheck(Protocol):
    def __call__(self) -> bool: ...


class WorkflowSynthesis(Protocol):
    def __call__(
        self,
        goal: GoalContract,
        actions: tuple[PlanStep, ...],
        observations: tuple[ToolResult, ...],
    ) -> str: ...


class WorkflowRevision(Protocol):
    def __call__(
        self,
        goal: GoalContract,
        actions: tuple[PlanStep, ...],
        observations: tuple[ToolResult, ...],
    ) -> PlanStep | None: ...


@dataclass(frozen=True)
class WorkflowRun:
    events: tuple[WorkflowEvent, ...]
    terminal: TerminalOutcome
    budget: BudgetSnapshot
    state: TurnState
    observations: tuple[ToolResult, ...] = ()


@dataclass
class _RunContext:
    state: TurnState
    ledger: BudgetLedger
    retries: RetryLedger
    stream: WorkflowEventStream
    results: list[ToolResult] = field(default_factory=list)


@dataclass
class _ActionExecution:
    reports: list[Verification] = field(default_factory=list)
    achieved_sources: set[SourceKind] = field(default_factory=set)
    required_verification_failed: bool = False
    planned_steps: list[PlanStep] = field(default_factory=list)
    executed_steps: list[PlanStep] = field(default_factory=list)
    failure: TerminalOutcome | None = None


@dataclass(frozen=True)
class _PlanPreparation:
    plan: ActionPlan | None = None
    failure: TerminalOutcome | None = None


@dataclass(frozen=True)
class _ActionProgress:
    failure: TerminalOutcome | None = None
    revised_step: PlanStep | None = None


class RetryLedger:
    def __init__(self) -> None:
        self._attempts: dict[str, int] = {}
        self._failures: dict[str, int] = {}
        self._completed: set[str] = set()

    def attempts(self, fingerprint: str) -> int:
        return self._attempts.get(fingerprint, 0)

    def failures(self, fingerprint: str) -> int:
        return self._failures.get(fingerprint, 0)

    def record_attempt(self, fingerprint: str) -> int:
        count = self.attempts(fingerprint) + 1
        self._attempts[fingerprint] = count
        return count

    def record_failure(self, fingerprint: str) -> int:
        count = self.failures(fingerprint) + 1
        self._failures[fingerprint] = count
        return count

    def mark_completed(self, fingerprint: str) -> None:
        self._completed.add(fingerprint)

    def is_completed(self, fingerprint: str) -> bool:
        return fingerprint in self._completed


def action_fingerprint(goal: GoalContract, call: ToolCall) -> str:
    canonical = json.dumps(
        {
            "goal_id": goal.goal_id,
            "goal_revision": {
                "objective": goal.objective,
                "constraints": [
                    {"kind": item.kind, "value": _jsonable(item.value)} for item in goal.constraints
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
    TurnNode.RESOLVE_GOAL: frozenset({TurnNode.CHOOSE_CAPABILITY}),
    TurnNode.CHOOSE_CAPABILITY: frozenset(
        {
            TurnNode.MAKE_PLAN,
            TurnNode.AUTHORIZE_ACTION,
            TurnNode.SYNTHESIZE,
            TurnNode.ASK_CLARIFICATION,
        }
    ),
    TurnNode.MAKE_PLAN: frozenset(
        {TurnNode.AUTHORIZE_ACTION, TurnNode.SYNTHESIZE, TurnNode.ASK_CLARIFICATION}
    ),
    TurnNode.AUTHORIZE_ACTION: frozenset({TurnNode.EXECUTE_ACTION}),
    TurnNode.EXECUTE_ACTION: frozenset({TurnNode.ASSESS_OBSERVATION}),
    TurnNode.ASSESS_OBSERVATION: frozenset(
        {TurnNode.AUTHORIZE_ACTION, TurnNode.REVISE_QUERY, TurnNode.SYNTHESIZE}
    ),
    TurnNode.REVISE_QUERY: frozenset({TurnNode.AUTHORIZE_ACTION}),
    TurnNode.SYNTHESIZE: frozenset({TurnNode.VERIFY_ANSWER}),
    TurnNode.VERIFY_ANSWER: frozenset({TurnNode.REPAIR_ANSWER, TurnNode.FINALIZE}),
    TurnNode.REPAIR_ANSWER: frozenset({TurnNode.VERIFY_ANSWER}),
    TurnNode.ASK_CLARIFICATION: frozenset({TurnNode.FINALIZE}),
}


@dataclass
class ControlledWorkflowRunner:
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
        initial_model_calls: int = 0,
        admission_error: str = "",
        synthesize: WorkflowSynthesis | None = None,
        revise: WorkflowRevision | None = None,
        allow_empty_plan: bool = False,
    ) -> WorkflowRun:
        if initial_model_calls < 0:
            raise ValueError("initial model calls must be non-negative")
        run_id = turn_id.strip() or f"run-{uuid4().hex}"
        ledger = BudgetLedger(self.budget)
        retries = RetryLedger()
        stream = WorkflowEventStream(
            session_id=session_id,
            run_id=run_id,
            goal_id=goal.goal_id,
            surface=surface,
        )
        state = TurnState(
            run_id=run_id,
            session_id=session_id,
            source=surface,
            goal=goal,
            node=TurnNode.ADMIT,
            budget=self.budget,
        )
        context = _RunContext(
            state=state,
            ledger=ledger,
            retries=retries,
            stream=stream,
        )
        def is_cancelled() -> bool:
            return bool(cancelled and cancelled())
        stream.emit(EventType.TURN_STARTED, context.state.node, status=EventStatus.STARTED)
        try:
            if admission_error:
                return self._terminal_run(
                    context,
                    self._failed_outcome(
                        "input_guardrail",
                        detail=admission_error,
                    )
                )
            if initial_model_calls:
                ledger.consume("model", initial_model_calls)
            if is_cancelled():
                return self._terminal_run(
                    context, self._cancelled_outcome("cancelled_before_start")
                )

            self._transition(context, TurnNode.RESOLVE_GOAL)
            stream.emit(
                EventType.GOAL_RESOLVED,
                context.state.node,
                status=EventStatus.COMPLETED,
                payload={"objective": goal.objective},
            )
            self._transition(context, TurnNode.CHOOSE_CAPABILITY)
            preparation = self._prepare_plan(
                context,
                goal,
                planner=planner,
                explicit_call=explicit_call,
                synthesize=synthesize,
                allow_empty_plan=allow_empty_plan,
            )
            if preparation.failure is not None:
                return self._terminal_run(context, preparation.failure)
            if preparation.plan is None:
                raise WorkflowError(
                    WorkflowErrorCode.PROTOCOL_ERROR,
                    "plan preparation returned neither plan nor failure",
                )

            execution = self._execute_actions(
                context,
                goal,
                preparation.plan,
                authorize=authorize,
                is_cancelled=is_cancelled,
                revise=revise,
            )
            if execution.failure is not None:
                return self._terminal_run(context, execution.failure)
            response = self._synthesize_and_verify(
                context,
                goal,
                synthesize=synthesize,
                execution=execution,
                is_cancelled=is_cancelled,
            )
            return self._terminal_run(context, response)
        except BudgetExceededError as exc:
            return self._terminal_run(
                context, self._failed_outcome("budget_exhausted", detail=exc.budget_name)
            )
        except InvalidTransitionError as exc:
            return self._terminal_run(
                context, self._failed_outcome("invalid_transition", detail=str(exc))
            )
        except PlanLookupError as exc:
            return self._terminal_run(context, self._failed_outcome(str(exc)))
        except PlanOutputError as exc:
            return self._terminal_run(
                context,
                self._failed_outcome(
                    "planner_output_invalid",
                    detail=exc.code,
                )
            )
        except WorkflowSynthesisError as exc:
            return self._terminal_run(
                context, self._failed_outcome("synthesis_failed", detail=str(exc))
            )
        except WorkflowRevisionError as exc:
            return self._terminal_run(
                context, self._failed_outcome(exc.reason_code, detail=str(exc))
            )
        except Exception as exc:  # noqa: BLE001 - controller boundary terminalizes faults
            return self._terminal_run(
                context,
                self._failed_outcome(
                    "workflow_error",
                    detail=type(exc).__name__,
                )
            )

    def _prepare_plan(
        self,
        context: _RunContext,
        goal: GoalContract,
        *,
        planner: WorkflowPlanner | None,
        explicit_call: ToolCall | None,
        synthesize: WorkflowSynthesis | None,
        allow_empty_plan: bool,
    ) -> _PlanPreparation:
        if explicit_call is not None:
            step = self._explicit_step(explicit_call)
            plan = ActionPlan(steps=(step,))
            context.stream.emit(
                EventType.STEP_COMPLETED,
                context.state.node,
                status=EventStatus.COMPLETED,
                payload={
                    "operation": "choose_capability",
                    "capability": step.capability.value,
                },
            )
        elif planner is None:
            if synthesize is None or not allow_empty_plan:
                return _PlanPreparation(failure=self._failed_outcome("planner_required"))
            plan = ActionPlan(steps=())
        else:
            context.stream.emit(
                EventType.STEP_COMPLETED,
                context.state.node,
                status=EventStatus.COMPLETED,
                payload={
                    "operation": "choose_capability",
                    "capability": "planner_catalog",
                },
            )
            self._transition(context, TurnNode.MAKE_PLAN)
            context.stream.emit(
                EventType.STEP_PROGRESS,
                context.state.node,
                payload={"operation": "plan"},
            )
            context.ledger.consume("planner")
            self._bind_planner_budget(planner, context.ledger)
            if not self._has_model_budget_hook(planner):
                context.ledger.consume("model")
            plan = planner.plan(goal.planner_text())

        if not plan.steps and (synthesize is None or not allow_empty_plan):
            return _PlanPreparation(failure=self._failed_outcome("empty_plan"))
        context.ledger.consume("planned_action", len(plan.steps))
        plan_node = TurnNode.AUTHORIZE_ACTION if plan.steps else TurnNode.SYNTHESIZE
        self._transition(
            context,
            plan_node,
            patch={"plan": tuple(step.as_planned_action() for step in plan.steps)},
        )
        if plan.public_update:
            context.stream.emit(
                EventType.PUBLIC_UPDATE,
                context.state.node,
                payload={
                    "text": plan.public_update,
                    "non_terminal": True,
                    "scheduled_actions": len(plan.steps),
                },
            )
        return _PlanPreparation(plan=plan)

    def _execute_actions(
        self,
        context: _RunContext,
        goal: GoalContract,
        plan: ActionPlan,
        *,
        authorize: AuthorizationPolicy | None,
        is_cancelled: Callable[[], bool],
        revise: WorkflowRevision | None,
    ) -> _ActionExecution:
        execution = _ActionExecution(planned_steps=list(plan.steps))
        index = 0
        while index < len(execution.planned_steps):
            if is_cancelled():
                execution.failure = self._cancelled_outcome("cancelled_before_action")
                return execution
            if index > 0:
                self._transition(context, TurnNode.AUTHORIZE_ACTION)
            progress = self._process_action(
                context,
                goal,
                execution,
                execution.planned_steps[index],
                authorize=authorize,
                is_cancelled=is_cancelled,
                revise=revise,
            )
            if progress.failure is not None:
                execution.failure = progress.failure
                return execution
            if progress.revised_step is not None:
                context.ledger.consume("planned_action")
                execution.planned_steps.append(progress.revised_step)
                self._transition(
                    context,
                    TurnNode.REVISE_QUERY,
                    patch={
                        "plan": tuple(
                            item.as_planned_action() for item in execution.planned_steps
                        ),
                    },
                )
                context.stream.emit(
                    EventType.STEP_PROGRESS,
                    context.state.node,
                    payload={
                        "operation": "revise_query",
                        "action_id": progress.revised_step.action_id,
                        "scheduled_actions": len(execution.planned_steps),
                    },
                )
            index += 1
        return execution

    def _process_action(
        self,
        context: _RunContext,
        goal: GoalContract,
        execution: _ActionExecution,
        step: PlanStep,
        *,
        authorize: AuthorizationPolicy | None,
        is_cancelled: Callable[[], bool],
        revise: WorkflowRevision | None,
    ) -> _ActionProgress:
        result, observation, verification, failure = self._run_step(
            context.state,
            step,
            ledger=context.ledger,
            retries=context.retries,
            stream=context.stream,
            authorize=authorize,
            is_cancelled=is_cancelled,
        )
        context.results.append(result)
        execution.executed_steps.append(step)
        revised_step = (
            revise(
                goal,
                tuple(execution.executed_steps),
                tuple(context.results),
            )
            if revise is not None
            else None
        )
        if step.required:
            if verification.achieved:
                execution.reports.append(verification)
                source_kind = source_for_capability(step.capability)
                if source_kind is not None:
                    execution.achieved_sources.add(source_kind)
            elif revised_step is None:
                # A failed read/search can be an intermediate observation. It
                # is not a terminal failure when the controller has scheduled
                # a typed refinement that may replace it with usable evidence.
                # Keep the failure terminal only when no refinement was
                # produced; otherwise the final verification is determined by
                # the effective action sequence below.
                execution.required_verification_failed = True
        context.state = replace(
            context.state,
            node=TurnNode.ASSESS_OBSERVATION,
            observations=context.state.observations + (observation,),
            verification=VerificationReport(
                passed=verification.achieved,
                reason_code=verification.reason,
                unmet_criteria=verification.unmet_criteria,
                supported_evidence_ids=verification.evidence_ids,
            ),
        )
        if failure is not None:
            return _ActionProgress(failure=failure)
        revisable = (
            revise is not None
            and result.ok
            and step.capability
            in {
                Capability.KNOWLEDGE_SEARCH,
                Capability.KNOWLEDGE_READ,
                Capability.MEMORY_SEARCH,
            }
        )
        if step.required and not verification.achieved and not revisable:
            return _ActionProgress(
                failure=self._failed_outcome(
                    verification.reason,
                    tool_status=result.status,
                    unmet_criteria=verification.unmet_criteria,
                )
            )

        return _ActionProgress(revised_step=revised_step)

    def _synthesize_and_verify(
        self,
        context: _RunContext,
        goal: GoalContract,
        *,
        synthesize: WorkflowSynthesis | None,
        execution: _ActionExecution,
        is_cancelled: Callable[[], bool],
    ) -> TerminalOutcome:
        if is_cancelled():
            return self._cancelled_outcome("cancelled_before_answer")
        if context.state.node is not TurnNode.SYNTHESIZE:
            self._transition(
                context,
                TurnNode.SYNTHESIZE,
                patch={"draft_answer": self._observation_text(context.results)},
            )
        context.stream.emit(
            EventType.STEP_STARTED,
            context.state.node,
            status=EventStatus.STARTED,
            payload={"operation": "synthesize"},
        )
        response_text = self._synthesis_text(
            synthesize,
            goal,
            tuple(execution.executed_steps),
            tuple(context.results),
            context.state.draft_answer or "",
        )
        if not response_text:
            return self._failed_outcome("empty_synthesis")
        citations = self._citation_payload(context.results)
        output_event = check_final_output(
            response_text,
            # A retrieval operation is not evidence by itself.  Empty or
            # degraded receipts must still reach synthesis so the model
            # can abstain explicitly instead of being replaced by a
            # generic workflow failure.
            citations=citations,
            knowledge_used=bool(citations),
            tool_results=tuple(context.results),
            policy=self.guardrail_policy,
        )
        if output_event.blocked:
            return self._failed_outcome("output_guardrail", detail=output_event.reason)

        self._transition(context, TurnNode.VERIFY_ANSWER)
        missing_sources = tuple(
            source.value
            for source in goal.required_sources
            if source not in execution.achieved_sources
        )
        unmet_success_criteria = unmet_goal_criteria(
            goal,
            achieved_sources=execution.achieved_sources,
            has_observation=any(report.achieved for report in execution.reports),
        )
        passed = (
            all(report.achieved for report in execution.reports)
            and not execution.required_verification_failed
            and not missing_sources
            and not unmet_success_criteria
        )
        evidence_ids = tuple(
            dict.fromkeys(
                evidence_id
                for report in execution.reports
                for evidence_id in report.evidence_ids
            )
        )
        final_report = VerificationReport(
            passed=passed,
            reason_code="goal_criteria_satisfied" if passed else "goal_criteria_unmet",
            unmet_criteria=tuple(
                criterion
                for report in execution.reports
                for criterion in report.unmet_criteria
            )
            + tuple(f"source:{source}" for source in missing_sources)
            + unmet_success_criteria,
            supported_evidence_ids=evidence_ids,
        )
        context.state = replace(context.state, verification=final_report)
        context.stream.emit(
            EventType.VERIFICATION_COMPLETED,
            context.state.node,
            status=EventStatus.COMPLETED if passed else EventStatus.FAILED,
            payload={"passed": passed, "evidence_ids": list(evidence_ids)},
        )
        if not passed:
            return self._failed_outcome(
                "goal_criteria_unmet",
                unmet_criteria=final_report.unmet_criteria,
            )
        self._transition(context, TurnNode.FINALIZE)
        return TerminalOutcome(
            status=TerminalStatus.ACHIEVED,
            final_text=response_text,
            goal_status=GoalStatus.ACHIEVED,
            evidence_ids=evidence_ids,
            route="controlled_workflow",
            metadata={
                "goal_id": goal.goal_id,
                "observations": len(context.results),
            },
        )

    def _transition(
        self,
        context: _RunContext,
        target: TurnNode,
        *,
        patch: Mapping[str, Any] | None = None,
    ) -> None:
        context.state = self._apply_advance(
            context.state,
            Advance(target, StatePatch(dict(patch or {}))),
            context.ledger,
            context.stream,
        )

    @staticmethod
    def _terminal_run(context: _RunContext, outcome: TerminalOutcome) -> WorkflowRun:
        context.state = replace(context.state, terminal=outcome)
        if context.stream.terminal_outcome is None:
            context.stream.emit_terminal(outcome)
        return WorkflowRun(
            events=tuple(context.stream),
            terminal=outcome,
            budget=context.ledger.snapshot(),
            state=context.state,
            observations=tuple(context.results),
        )

    def _run_step(
        self,
        state: TurnState,
        step: PlanStep,
        *,
        ledger: BudgetLedger,
        retries: RetryLedger,
        stream: WorkflowEventStream,
        authorize: AuthorizationPolicy | None,
        is_cancelled: Callable[[], bool],
    ) -> tuple[ToolResult, Observation, Verification, TerminalOutcome | None]:
        fingerprint = action_fingerprint(state.goal, step.call)
        tool = self.tool_runtime.get(step.call.name)
        if tool is None or not tool.spec.enabled:
            result = ToolResult(
                step.call.name,
                False,
                "",
                error="unknown_tool",
                status=ToolExecutionStatus.INVALID,
            )
            return (
                result,
                self._observation(step, result),
                Verification(False, "unknown_tool"),
                self._failed_outcome("unknown_tool") if step.required else None,
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
                status=ToolExecutionStatus.DENIED,
            )
            return (
                result,
                self._observation(step, result),
                Verification(False, guardrail_event.reason),
                (
                    self._failed_outcome(
                        "guardrail_blocked",
                        detail=guardrail_event.reason,
                    )
                    if step.required
                    else None
                ),
            )
        needs_authorization = (
            step.requires_authorization or tool.spec.side_effect != SideEffectLevel.READ_ONLY
        )
        if needs_authorization and (authorize is None or not authorize(state.goal, step)):
            result = ToolResult(
                step.call.name,
                False,
                "",
                error="authorization_denied",
                status=ToolExecutionStatus.DENIED,
            )
            return (
                result,
                self._observation(step, result),
                Verification(False, "authorization_denied"),
                (self._failed_outcome("authorization_denied") if step.required else None),
            )

        if retries.is_completed(fingerprint):
            result = ToolResult(
                step.call.name,
                False,
                "",
                error="duplicate_action",
                status=ToolExecutionStatus.DENIED,
            )
            return (
                result,
                self._observation(step, result),
                Verification(False, "duplicate_action"),
                self._failed_outcome("duplicate_action") if step.required else None,
            )

        while True:
            if is_cancelled():
                result = ToolResult(
                    step.call.name,
                    False,
                    "",
                    error="cancelled",
                    status=ToolExecutionStatus.CANCELLED,
                )
                return (
                    result,
                    self._observation(step, result),
                    Verification(False, "cancelled"),
                    self._cancelled_outcome("cancelled_during_action"),
                )
            stream.emit(
                EventType.STEP_STARTED,
                TurnNode.AUTHORIZE_ACTION,
                status=EventStatus.STARTED,
                payload={"action_id": step.action_id, "operation": "authorize"},
            )
            ledger.consume("transition")
            stream.emit(
                EventType.STEP_STARTED,
                TurnNode.EXECUTE_ACTION,
                status=EventStatus.STARTED,
                payload={"action_id": step.action_id, "operation": "execute"},
            )
            ledger.consume("tool")
            retries.record_attempt(fingerprint)
            result = ToolExecutionNode(self.tool_runtime).execute(step.call)
            if not isinstance(result, ToolResult):
                raise TypeError("tool runtime returned an invalid result")
            if result.ok:
                retries.mark_completed(fingerprint)
            else:
                retries.record_failure(fingerprint)
            ledger.consume("transition")
            stream.emit(
                EventType.STEP_COMPLETED,
                TurnNode.ASSESS_OBSERVATION,
                status=EventStatus.COMPLETED if result.ok else EventStatus.FAILED,
                payload={
                    "action_id": step.action_id,
                    "result_status": cast(ToolExecutionStatus, result.status).value,
                    "retryable": result.retryable,
                },
            )
            action = step.as_planned_action()
            verification = self.verifier.verify(state.goal, result, action)
            if verification.achieved:
                return result, self._observation(step, result), verification, None
            retry_allowed = (
                result.retryable
                and (tool.spec.side_effect == SideEffectLevel.READ_ONLY or tool.spec.idempotent)
                and retries.failures(fingerprint) < self.budget.max_same_action_failures
            )
            if retry_allowed:
                ledger.consume("retry")
                stream.emit(
                    EventType.STEP_PROGRESS,
                    TurnNode.ASSESS_OBSERVATION,
                    payload={
                        "action_id": step.action_id,
                        "operation": "retry",
                        "failure_count": retries.failures(fingerprint),
                    },
                )
                continue
            return result, self._observation(step, result), verification, None

    def _explicit_step(self, call: ToolCall) -> PlanStep:
        tool = self.tool_runtime.get(call.name)
        if tool is None or not tool.spec.enabled:
            raise PlanLookupError("unknown_tool")
        try:
            capability = Capability(tool.spec.workflow_capability)
        except ValueError as exc:
            raise PlanLookupError("unsupported_tool_capability") from exc
        return PlanStep(
            action_id="explicit-1",
            capability=capability,
            call=call,
            purpose="explicit_command",
            expected_observation="tool receipt",
            requires_authorization=tool.spec.side_effect != SideEffectLevel.READ_ONLY,
        )

    @staticmethod
    def _observation(step: PlanStep, result: ToolResult) -> Observation:
        status = cast(ToolExecutionStatus, result.status)
        data = dict(result.data)
        if result.content:
            data["content"] = result.content
        return Observation(
            action_id=step.action_id,
            status=status,
            data=data,
            error_code=tool_error_code(result) if not result.ok else None,
            retryable=result.retryable,
            committed=result.ok,
            receipt=action_fingerprint_stub(step, result),
        )

    @staticmethod
    def _apply_advance(
        state: TurnState,
        outcome: Advance,
        ledger: BudgetLedger,
        stream: WorkflowEventStream,
    ) -> TurnState:
        if outcome.next_node not in _TRANSITIONS.get(state.node, frozenset()):
            raise InvalidTransitionError(state.node, outcome.next_node)
        ledger.consume("transition")
        patch = dict(outcome.patch.values)
        unknown = set(patch) - set(TurnState.__dataclass_fields__)
        if unknown:
            raise WorkflowError(
                WorkflowErrorCode.PROTOCOL_ERROR,
                "state patch contains unknown fields",
            )
        now = datetime.now(UTC).isoformat()
        trace = state.trace + (
            NodeTrace(
                node=outcome.next_node,
                started_at=now,
                finished_at=now,
            ),
        )
        stream.emit(
            EventType.STEP_STARTED,
            outcome.next_node,
            status=EventStatus.STARTED,
            payload={"operation": outcome.next_node.value},
        )
        return replace(
            state,
            node=outcome.next_node,
            trace=trace,
            **patch,
        )

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
    def _synthesis_text(
        synthesize: WorkflowSynthesis | None,
        goal: GoalContract,
        actions: tuple[PlanStep, ...],
        observations: tuple[ToolResult, ...],
        default_text: str,
    ) -> str:
        if synthesize is None:
            return default_text
        return synthesize(goal, actions, observations).strip()

    @staticmethod
    def _citation_payload(
        observations: list[ToolResult],
    ) -> tuple[object, ...]:
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
    def _bind_planner_budget(
        planner: WorkflowPlanner,
        ledger: BudgetLedger,
    ) -> None:
        hooks = getattr(planner, "set_budget_hooks", None)
        if callable(hooks):
            hooks(
                model_call=lambda: ledger.consume("model"),
                structured_repair=lambda: ledger.consume("structured_repair"),
            )
            return
        hook = getattr(planner, "set_model_call_hook", None)
        if callable(hook):
            hook(lambda: ledger.consume("model"))

    @staticmethod
    def _has_model_budget_hook(planner: WorkflowPlanner) -> bool:
        return callable(getattr(planner, "set_budget_hooks", None)) or callable(
            getattr(planner, "set_model_call_hook", None)
        )

    @staticmethod
    def _failed_outcome(
        code: str,
        *,
        detail: str = "",
        tool_status: ToolExecutionStatus | None = None,
        unmet_criteria: tuple[str, ...] = (),
    ) -> TerminalOutcome:
        metadata: dict[str, Any] = {}
        if detail:
            metadata["detail"] = detail
        if code == "budget_exhausted":
            terminal_status = TerminalStatus.BUDGET_EXHAUSTED
        elif code == "cancelled" or tool_status is ToolExecutionStatus.CANCELLED:
            terminal_status = TerminalStatus.CANCELLED
        elif tool_status is ToolExecutionStatus.NOT_FOUND or code in {
            "no_matching_observation",
            "required_source_not_used",
            "expected_observation_missing",
            "goal_criteria_unmet",
            "tool_returned_no_observation",
            "not_found",
            "tool_not_found",
        }:
            terminal_status = TerminalStatus.INSUFFICIENT_EVIDENCE
        elif tool_status in {
            ToolExecutionStatus.INVALID,
            ToolExecutionStatus.DENIED,
        } or code in {
            "authorization_denied",
            "duplicate_action",
            "guardrail_blocked",
            "output_guardrail",
            "input_guardrail",
            "unknown_tool",
            "unsupported_tool_capability",
            "planner_output_invalid",
            "forbidden_plan_fields",
            "invalid_tool_input",
            "empty_plan",
            "planner_required",
        }:
            terminal_status = TerminalStatus.SAFE_FAILURE
        else:
            terminal_status = TerminalStatus.SYSTEM_FAILURE
        return TerminalOutcome(
            status=terminal_status,
            final_text="",
            goal_status=GoalStatus.FAILED,
            unmet_criteria=unmet_criteria,
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


class PlanLookupError(ValueError):
    pass


def action_fingerprint_stub(step: PlanStep, result: ToolResult) -> str:
    payload = json.dumps(
        {
            "action_id": step.action_id,
            "tool": step.call.name,
            "status": cast(ToolExecutionStatus, result.status).value,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
