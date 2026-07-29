from __future__ import annotations

from dataclasses import dataclass

from soca.core.usage import LLMUsage, SessionUsage, TurnUsage


@dataclass
class FakeLLMResult:
    n_prompt_tokens: int = 120
    n_completion_tokens: int = 40
    ttft_ms: float = 80.0
    total_latency_ms: float = 700.0
    tokens_per_second: float = 60.0


@dataclass(frozen=True)
class FakeRoute:
    value: str


@dataclass
class FakeTrace:
    used_tool: bool = False
    used_llm: bool = True
    stage_latencies_ms: dict | None = None


@dataclass
class FakeResult:
    route: FakeRoute
    blocked: bool = False
    trace: FakeTrace | None = None
    usage: LLMUsage | None = None


def test_llm_usage_from_result_maps_fields() -> None:
    usage = LLMUsage.from_llm_result(FakeLLMResult())
    assert usage == LLMUsage(
        prompt_tokens=120,
        completion_tokens=40,
        ttft_ms=80.0,
        total_latency_ms=700.0,
        tokens_per_second=60.0,
    )


def test_llm_usage_from_none_is_none() -> None:
    assert LLMUsage.from_llm_result(None) is None


def test_llm_usage_combines_sequential_calls() -> None:
    first = LLMUsage(
        prompt_tokens=100,
        completion_tokens=20,
        ttft_ms=50.0,
        total_latency_ms=500.0,
        tokens_per_second=50.0,
    )
    repair = LLMUsage(
        prompt_tokens=80,
        completion_tokens=10,
        ttft_ms=40.0,
        total_latency_ms=300.0,
        tokens_per_second=50.0,
    )

    combined = first.combine(repair)

    assert combined.prompt_tokens == 180
    assert combined.completion_tokens == 30
    assert combined.total_latency_ms == 800.0
    assert combined.ttft_ms == 540.0
    assert combined.tokens_per_second == 50.0


def test_turn_usage_from_runtime_result_reads_route_and_llm() -> None:
    llm = LLMUsage.from_llm_result(FakeLLMResult())
    result = FakeResult(
        route=FakeRoute("free_chat"),
        trace=FakeTrace(used_llm=True, stage_latencies_ms={"llm": 530.0}),
        usage=llm,
    )

    turn = TurnUsage.from_runtime_result(result)

    assert turn.route == "free_chat"
    assert turn.used_llm is True
    assert turn.llm == llm
    assert turn.runtime_latency_ms == 530.0


def test_turn_usage_tool_route_has_no_llm() -> None:
    result = FakeResult(
        route=FakeRoute("tool_direct"),
        trace=FakeTrace(used_tool=True, used_llm=False, stage_latencies_ms={"tool:x": 3.0}),
        usage=None,
    )

    turn = TurnUsage.from_runtime_result(result)

    assert turn.route == "tool_direct"
    assert turn.used_tool is True
    assert turn.llm is None


def test_session_usage_aggregates_immutably() -> None:
    llm = LLMUsage(prompt_tokens=100, completion_tokens=30, ttft_ms=80.0, tokens_per_second=60.0)
    other = LLMUsage(prompt_tokens=50, completion_tokens=20, ttft_ms=40.0, tokens_per_second=40.0)

    session = SessionUsage()
    s1 = session.add(TurnUsage(route="free_chat", used_llm=True, llm=llm))
    s2 = s1.add(TurnUsage(route="tool_direct", used_tool=True))  # no llm
    s3 = s2.add(TurnUsage(route="free_chat", used_llm=True, llm=other))

    # original stays empty (immutability)
    assert session.total_turns == 0
    assert s3.total_turns == 3
    assert s3.llm_turns == 2
    assert s3.total_prompt_tokens == 150
    assert s3.total_completion_tokens == 50
    assert s3.mean_tokens_per_second == 50.0  # (60 + 40) / 2
