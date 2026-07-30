from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from soca.core import AssistantRuntime, RuntimeOptions
from soca.core.workflow import (
    ActionPlan,
    ControlledWorkflowRunner,
    GoalConstraint,
    GoalContract,
    GoalResolver,
    PlanStep,
    TerminalStatus,
    TurnBudget,
    action_fingerprint,
)
from soca.tools import (
    SideEffectLevel,
    ToolCall,
    ToolResult,
    ToolRuntime,
    ToolSpec,
    object_schema,
)


@dataclass
class ScriptedTool:
    responses: list[ToolResult]
    name: str = "knowledge.search"
    side_effect: SideEffectLevel = SideEffectLevel.READ_ONLY
    calls: int = 0

    @property
    def spec(self) -> ToolSpec:
        properties = (
            {"path": {"type": "string"}}
            if self.name == "knowledge.read"
            else {"query": {"type": "string"}}
        )
        required = ["path"] if self.name == "knowledge.read" else ["query"]
        return ToolSpec(
            name=self.name,
            description="Search a test knowledge source.",
            input_schema=object_schema(
                properties=properties,
                required=required,
            ),
            side_effect=self.side_effect,
        )

    def run(self, arguments: dict) -> ToolResult:
        del arguments
        self.calls += 1
        if self.responses:
            return self.responses.pop(0)
        return ToolResult(self.name, ok=True, content="observation")


@dataclass
class StaticPlanner:
    plan_value: ActionPlan
    calls: int = 0
    seen_goals: list[str] = field(default_factory=list)

    def plan(self, goal: str) -> ActionPlan:
        assert goal
        self.calls += 1
        self.seen_goals.append(goal)
        return self.plan_value


@dataclass
class RepairLLM:
    responses: list[str]
    max_tokens_seen: list[int] = field(default_factory=list)

    def generate(
        self,
        user_msg: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        inject_persona: bool = True,
    ):
        del user_msg, temperature, top_p, inject_persona
        self.max_tokens_seen.append(max_tokens)
        from soca.llm import LLMResult

        return LLMResult(
            text=self.responses.pop(0),
            prompt="planner",
            n_prompt_tokens=1,
            n_completion_tokens=1,
            ttft_ms=0,
            total_latency_ms=0,
            tokens_per_second=1,
        )


def make_goal() -> GoalContract:
    return GoalContract(goal_id="goal-1", objective="Tìm ghi chú Bayes")


def knowledge_observation(
    content: str,
    *,
    ok: bool = True,
    error: str = "",
) -> ToolResult:
    return ToolResult(
        "knowledge.search",
        ok,
        content,
        data={"hits": [{"path": "wiki/bayes.md"}]} if ok else {},
        error=error,
    )


def make_plan(*calls: ToolCall) -> ActionPlan:
    return ActionPlan(
        steps=tuple(
            PlanStep(
                action_id=f"action-{index}",
                call=call,
                purpose="retrieve evidence",
            )
            for index, call in enumerate(calls, start=1)
        ),
        final_instruction="Đã tìm thấy bằng chứng.",
        rationale="read-only retrieval",
    )


def test_explicit_call_skips_planner_and_emits_update_before_terminal() -> None:
    tool = ScriptedTool([knowledge_observation("Bayes")])
    planner = StaticPlanner(make_plan(ToolCall("knowledge.search", {"query": "Bayes"})))
    runner = ControlledWorkflowRunner(ToolRuntime([tool]))

    result = runner.run(
        make_goal(),
        planner=planner,
        explicit_call=ToolCall("knowledge.search", {"query": "Bayes"}),
    )

    assert result.terminal.status is TerminalStatus.ACHIEVED
    assert planner.calls == 0
    assert tool.calls == 1
    assert result.events[-1].terminal is True
    assert any(not event.terminal for event in result.events[:-1])
    assert sum(event.terminal for event in result.events) == 1


def test_planner_workflow_executes_catalog_action() -> None:
    tool = ScriptedTool([knowledge_observation("Bayes")])
    planner = StaticPlanner(make_plan(ToolCall("knowledge.search", {"query": "Bayes"})))

    result = ControlledWorkflowRunner(ToolRuntime([tool])).run(
        make_goal(),
        planner=planner,
    )

    assert result.terminal.status is TerminalStatus.ACHIEVED
    assert result.terminal.route == "controlled_workflow"
    assert planner.calls == 1
    assert result.observations[0].content == "Bayes"


def test_four_default_tool_calls_fit_transition_budget() -> None:
    tool = ScriptedTool([knowledge_observation(f"hit-{index}") for index in range(4)])
    calls = tuple(
        ToolCall("knowledge.search", {"query": f"query-{index}"})
        for index in range(4)
    )

    result = ControlledWorkflowRunner(
        ToolRuntime([tool]),
        budget=TurnBudget(max_transitions=24),
    ).run(
        make_goal(),
        planner=StaticPlanner(make_plan(*calls)),
    )

    assert result.terminal.status is TerminalStatus.ACHIEVED
    assert result.budget.transitions <= result.terminal.metadata.get(
        "transition_limit", result.budget.transitions
    )
    assert result.budget.tool_calls == 4


def test_transient_tool_failure_retries_with_shared_budget() -> None:
    tool = ScriptedTool(
        [
            knowledge_observation("", ok=False, error="temporary"),
            knowledge_observation("Bayes"),
        ]
    )
    plan = make_plan(ToolCall("knowledge.search", {"query": "Bayes"}))

    result = ControlledWorkflowRunner(
        ToolRuntime([tool]),
        budget=TurnBudget(max_readonly_tool_retries=1, max_tool_calls=2),
    ).run(make_goal(), planner=StaticPlanner(plan))

    assert result.terminal.status is TerminalStatus.ACHIEVED
    assert tool.calls == 2
    assert result.budget.retries == 1
    assert any(event.payload.get("operation") == "retry" for event in result.events)


def test_mutating_tool_failure_is_not_retried_without_idempotency() -> None:
    tool = ScriptedTool(
        [
            ToolResult("memory.propose_note", False, "", error="ambiguous"),
            ToolResult("memory.propose_note", True, "proposal"),
        ],
        name="memory.propose_note",
        side_effect=SideEffectLevel.LOCAL_STATE,
    )
    result = ControlledWorkflowRunner(
        ToolRuntime([tool]),
        budget=TurnBudget(max_readonly_tool_retries=1),
    ).run(
        make_goal(),
        explicit_call=ToolCall("memory.propose_note", {"query": "remember"}),
        authorize=lambda goal, step: True,
    )

    assert result.terminal.status is TerminalStatus.SYSTEM_FAILURE
    assert result.terminal.error_code == "ambiguous"
    assert result.budget.retries == 0
    assert tool.calls == 1


def test_duplicate_successful_action_is_gated() -> None:
    tool = ScriptedTool(
        [
            knowledge_observation("first"),
            knowledge_observation("second"),
        ]
    )
    call = ToolCall("knowledge.search", {"query": "Bayes"})
    result = ControlledWorkflowRunner(ToolRuntime([tool])).run(
        make_goal(),
        planner=StaticPlanner(make_plan(call, call)),
    )

    assert result.terminal.status is TerminalStatus.SAFE_FAILURE
    assert result.terminal.error_code == "duplicate_action"
    assert tool.calls == 1


def test_empty_search_observation_does_not_achieve_goal() -> None:
    tool = ScriptedTool(
        [ToolResult("knowledge.search", True, "Không tìm thấy", data={"hits": []})]
    )
    result = ControlledWorkflowRunner(ToolRuntime([tool])).run(
        make_goal(),
        explicit_call=ToolCall("knowledge.search", {"query": "missing"}),
    )

    assert result.terminal.status is TerminalStatus.INSUFFICIENT_EVIDENCE
    assert result.terminal.error_code == "no_matching_observation"


def test_knowledge_read_guardrail_blocks_out_of_scope_path_before_tool() -> None:
    tool = ScriptedTool(
        [ToolResult("knowledge.read", True, "secret")],
        name="knowledge.read",
    )
    result = ControlledWorkflowRunner(ToolRuntime([tool])).run(
        make_goal(),
        explicit_call=ToolCall("knowledge.read", {"path": "private/secret.md"}),
    )

    assert result.terminal.status is TerminalStatus.SAFE_FAILURE
    assert result.terminal.error_code == "guardrail_blocked"
    assert tool.calls == 0


def test_budget_exhaustion_is_terminal_and_bounded() -> None:
    tool = ScriptedTool([knowledge_observation("Bayes")])
    result = ControlledWorkflowRunner(
        ToolRuntime([tool]),
        budget=TurnBudget(max_tool_calls=0),
    ).run(
        make_goal(),
        explicit_call=ToolCall("knowledge.search", {"query": "Bayes"}),
    )

    assert result.terminal.status is TerminalStatus.BUDGET_EXHAUSTED
    assert result.terminal.error_code == "budget_exhausted"
    assert tool.calls == 0
    assert result.events[-1].terminal is True


def test_cancellation_does_not_produce_a_success_answer() -> None:
    tool = ScriptedTool([knowledge_observation("Bayes")])
    result = ControlledWorkflowRunner(ToolRuntime([tool])).run(
        make_goal(),
        explicit_call=ToolCall("knowledge.search", {"query": "Bayes"}),
        cancelled=lambda: True,
    )

    assert result.terminal.status is TerminalStatus.CANCELLED
    assert result.terminal.final_text == ""
    assert tool.calls == 0


def test_side_effect_action_requires_authorization() -> None:
    tool = ScriptedTool(
        [ToolResult("memory.propose_note", True, "proposal")],
        name="memory.propose_note",
        side_effect=SideEffectLevel.LOCAL_STATE,
    )
    call = ToolCall("memory.propose_note", {"query": "remember this"})
    result = ControlledWorkflowRunner(ToolRuntime([tool])).run(
        make_goal(),
        explicit_call=call,
    )

    assert result.terminal.status is TerminalStatus.SAFE_FAILURE
    assert result.terminal.error_code == "authorization_denied"
    assert tool.calls == 0


def test_action_fingerprint_is_stable_and_goal_scoped() -> None:
    call = ToolCall("knowledge.search", {"query": "Bayes"})
    same = action_fingerprint(make_goal(), call)
    again = action_fingerprint(make_goal(), call)
    other_goal = action_fingerprint(
        GoalContract(goal_id="goal-2", objective="Tìm ghi chú ONNX"),
        call,
    )

    assert same == again
    assert same != other_goal


def test_goal_resolver_keeps_follow_up_on_the_same_goal() -> None:
    resolver = GoalResolver()
    first = resolver.resolve("Tìm ghi chú Bayes")
    follow_up = resolver.resolve("chỉ lấy trong vault", continues_active_goal=True)

    assert follow_up.continued is True
    assert follow_up.goal.goal_id == first.goal.goal_id
    assert follow_up.goal.objective == first.goal.objective
    assert follow_up.goal.constraints[-1].kind == "follow_up"
    assert follow_up.goal.constraints[-1].value == "chỉ lấy trong vault"


def test_follow_up_constraint_is_sent_to_planner() -> None:
    goal = GoalContract(
        goal_id="goal-1",
        objective="Tìm ghi chú Bayes",
        constraints=(GoalConstraint("follow_up", "chỉ lấy trong vault"),),
    )
    planner = StaticPlanner(make_plan(ToolCall("knowledge.search", {"query": "Bayes"})))
    tool = ScriptedTool([knowledge_observation("Bayes")])

    ControlledWorkflowRunner(ToolRuntime([tool])).run(goal, planner=planner)

    assert planner.seen_goals == [
        "Objective: Tìm ghi chú Bayes\nConstraints: follow_up=chỉ lấy trong vault"
    ]


def test_structured_planner_repairs_once_and_debits_each_model_call() -> None:
    runtime = ToolRuntime([ScriptedTool([])])
    valid = (
        '{"steps":[{"action_id":"a1","tool":"knowledge.search",'
        '"arguments":{"query":"Bayes"},"purpose":"retrieve",'
        '"requires_authorization":false}],"final_instruction":"ok",'
        '"rationale":"evidence"}'
    )
    from soca.core.workflow import StructuredWorkflowPlanner

    planner = StructuredWorkflowPlanner(
        RepairLLM(["not json", valid]),
        runtime,
        repair_attempts=1,
    )
    calls = 0

    def debit() -> None:
        nonlocal calls
        calls += 1

    planner.set_model_call_hook(debit)
    plan = planner.plan("Tìm ghi chú Bayes")

    assert plan.steps[0].call.name == "knowledge.search"
    assert calls == 2


def test_structured_planner_clamps_output_to_model_capability() -> None:
    runtime = ToolRuntime([ScriptedTool([])])
    valid = (
        '{"steps":[],"final_instruction":"ok",'
        '"rationale":"no action"}'
    )
    llm = RepairLLM([valid])
    from soca.core.workflow import StructuredWorkflowPlanner

    planner = StructuredWorkflowPlanner(
        llm,
        runtime,
        max_tokens=256,
        model_context_window=2_048,
        model_max_output_tokens=64,
    )

    planner.plan("Trả lời ngắn gọn")

    assert llm.max_tokens_seen == [64]
    assert planner.last_prompt_manifest is not None
    assert planner.last_prompt_manifest["provider_prompt_tokens"] == 1


def test_structured_planner_blocks_known_context_overflow_before_model_call() -> None:
    runtime = ToolRuntime([ScriptedTool([])])
    llm = RepairLLM([])
    from soca.core.workflow import PlanOutputError, StructuredWorkflowPlanner

    planner = StructuredWorkflowPlanner(
        llm,
        runtime,
        model_context_window=64,
    )

    with pytest.raises(PlanOutputError, match="context_budget_exceeded"):
        planner.plan("Tìm ghi chú")
    assert llm.max_tokens_seen == []


def test_runner_budget_covers_planner_repair_calls() -> None:
    runtime = ToolRuntime([ScriptedTool([])])
    valid = (
        '{"steps":[{"action_id":"a1","tool":"knowledge.search",'
        '"arguments":{"query":"Bayes"},"purpose":"retrieve",'
        '"requires_authorization":false}],"final_instruction":"ok",'
        '"rationale":"evidence"}'
    )
    from soca.core.workflow import StructuredWorkflowPlanner

    planner = StructuredWorkflowPlanner(RepairLLM(["not json", valid]), runtime)
    result = ControlledWorkflowRunner(
        runtime,
        budget=TurnBudget(max_model_calls=1),
    ).run(make_goal(), planner=planner)

    assert result.terminal.status is TerminalStatus.BUDGET_EXHAUSTED
    assert result.terminal.error_code == "budget_exhausted"
    assert result.budget.model_calls == 1


def test_planner_catalog_does_not_invent_removed_weather_tool() -> None:
    runtime = ToolRuntime([ScriptedTool([])])
    from soca.core.workflow.planner import plan_schema

    schema = plan_schema(runtime)
    schema_text = str(schema)

    assert "weather" not in schema_text


def test_runtime_facade_is_opt_in_and_uses_active_goal_store() -> None:
    tool = ScriptedTool([knowledge_observation("Bayes")])
    runtime = AssistantRuntime(
        tool_runtime=ToolRuntime([tool]),
        options=RuntimeOptions(turn_workflow="shadow"),
    )

    result = runtime.run_controlled_workflow(
        "Tìm ghi chú Bayes",
        explicit_call=ToolCall("knowledge.search", {"query": "Bayes"}),
    )

    assert result.terminal.status is TerminalStatus.ACHIEVED
    assert runtime.options.turn_workflow == "shadow"
