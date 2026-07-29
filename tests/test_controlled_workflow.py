from __future__ import annotations

from dataclasses import dataclass

from soca.core import AssistantRuntime, RuntimeOptions
from soca.core.workflow import (
    ActionPlan,
    ControlledWorkflowRunner,
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
        return ToolSpec(
            name=self.name,
            description="Search a test knowledge source.",
            input_schema=object_schema(
                properties={"query": {"type": "string"}},
                required=["query"],
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

    def plan(self, goal: str) -> ActionPlan:
        assert goal
        self.calls += 1
        return self.plan_value


@dataclass
class RepairLLM:
    responses: list[str]

    def generate(
        self,
        user_msg: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        inject_persona: bool = True,
    ):
        del user_msg, max_tokens, temperature, top_p, inject_persona
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
    return GoalContract(statement="Tìm ghi chú Bayes", goal_id="goal-1")


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
    tool = ScriptedTool([ToolResult("knowledge.search", True, "Bayes")])
    planner = StaticPlanner(make_plan(ToolCall("knowledge.search", {"query": "Bayes"})))
    runner = ControlledWorkflowRunner(ToolRuntime([tool]))

    result = runner.run(
        make_goal(),
        planner=planner,
        explicit_call=ToolCall("knowledge.search", {"query": "Bayes"}),
    )

    assert result.terminal.status is TerminalStatus.SUCCEEDED
    assert planner.calls == 0
    assert tool.calls == 1
    assert result.events[-1].terminal is True
    assert any(event.kind == "update" for event in result.events[:-1])
    assert [event.kind for event in result.events].count("terminal") == 1


def test_planner_workflow_executes_catalog_action() -> None:
    tool = ScriptedTool([ToolResult("knowledge.search", True, "Bayes")])
    planner = StaticPlanner(make_plan(ToolCall("knowledge.search", {"query": "Bayes"})))

    result = ControlledWorkflowRunner(ToolRuntime([tool])).run(
        make_goal(),
        planner=planner,
    )

    assert result.terminal.status is TerminalStatus.SUCCEEDED
    assert result.terminal.route == "controlled_workflow"
    assert planner.calls == 1
    assert result.observations[0].content == "Bayes"


def test_transient_tool_failure_retries_with_shared_budget() -> None:
    tool = ScriptedTool(
        [
            ToolResult("knowledge.search", False, "", error="temporary"),
            ToolResult("knowledge.search", True, "Bayes"),
        ]
    )
    plan = make_plan(ToolCall("knowledge.search", {"query": "Bayes"}))

    result = ControlledWorkflowRunner(
        ToolRuntime([tool]),
        budget=TurnBudget(max_retries=1, max_tool_calls=2),
    ).run(make_goal(), planner=StaticPlanner(plan))

    assert result.terminal.status is TerminalStatus.SUCCEEDED
    assert tool.calls == 2
    assert result.budget.retries == 1
    assert any(event.payload.get("phase") == "retrying" for event in result.events)


def test_duplicate_successful_action_is_gated() -> None:
    tool = ScriptedTool(
        [
            ToolResult("knowledge.search", True, "first"),
            ToolResult("knowledge.search", True, "second"),
        ]
    )
    call = ToolCall("knowledge.search", {"query": "Bayes"})
    result = ControlledWorkflowRunner(ToolRuntime([tool])).run(
        make_goal(),
        planner=StaticPlanner(make_plan(call, call)),
    )

    assert result.terminal.status is TerminalStatus.FAILED
    assert result.terminal.error_code == "duplicate_action"
    assert tool.calls == 1


def test_budget_exhaustion_is_terminal_and_bounded() -> None:
    tool = ScriptedTool([ToolResult("knowledge.search", True, "Bayes")])
    result = ControlledWorkflowRunner(
        ToolRuntime([tool]),
        budget=TurnBudget(max_tool_calls=0),
    ).run(
        make_goal(),
        explicit_call=ToolCall("knowledge.search", {"query": "Bayes"}),
    )

    assert result.terminal.status is TerminalStatus.FAILED
    assert result.terminal.error_code == "budget_exhausted"
    assert tool.calls == 0
    assert result.events[-1].terminal is True


def test_cancellation_does_not_produce_a_success_answer() -> None:
    tool = ScriptedTool([ToolResult("knowledge.search", True, "Bayes")])
    result = ControlledWorkflowRunner(ToolRuntime([tool])).run(
        make_goal(),
        explicit_call=ToolCall("knowledge.search", {"query": "Bayes"}),
        cancelled=lambda: True,
    )

    assert result.terminal.status is TerminalStatus.CANCELLED
    assert result.terminal.response_text == ""
    assert tool.calls == 0


def test_side_effect_action_requires_authorization() -> None:
    tool = ScriptedTool(
        [ToolResult("memory.propose_note", True, "proposal")],
        name="memory.propose_note",
        side_effect=SideEffectLevel.LOCAL_STATE,
    )
    call = ToolCall("memory.propose_note", {"content": "remember this"})
    result = ControlledWorkflowRunner(ToolRuntime([tool])).run(
        make_goal(),
        explicit_call=call,
    )

    assert result.terminal.status is TerminalStatus.FAILED
    assert result.terminal.error_code == "authorization_denied"
    assert tool.calls == 0


def test_action_fingerprint_is_stable_and_goal_scoped() -> None:
    call = ToolCall("knowledge.search", {"query": "Bayes"})
    same = action_fingerprint(make_goal(), call)
    again = action_fingerprint(make_goal(), call)
    other_goal = action_fingerprint(
        GoalContract(statement="Tìm ghi chú ONNX", goal_id="goal-2"),
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
    assert follow_up.goal.statement == first.goal.statement
    assert follow_up.goal.metadata["follow_up"] == "chỉ lấy trong vault"


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

    assert result.terminal.status is TerminalStatus.FAILED
    assert result.terminal.error_code == "budget_exhausted"
    assert result.budget.model_calls == 1


def test_planner_catalog_does_not_invent_removed_weather_tool() -> None:
    runtime = ToolRuntime([ScriptedTool([])])
    from soca.core.workflow.planner import plan_schema

    schema = plan_schema(runtime)
    schema_text = str(schema)

    assert "weather" not in schema_text


def test_runtime_facade_is_opt_in_and_uses_active_goal_store() -> None:
    tool = ScriptedTool([ToolResult("knowledge.search", True, "Bayes")])
    runtime = AssistantRuntime(
        tool_runtime=ToolRuntime([tool]),
        options=RuntimeOptions(turn_workflow="shadow"),
    )

    result = runtime.run_controlled_workflow(
        "Tìm ghi chú Bayes",
        explicit_call=ToolCall("knowledge.search", {"query": "Bayes"}),
    )

    assert result.terminal.status is TerminalStatus.SUCCEEDED
    assert runtime.options.turn_workflow == "shadow"
