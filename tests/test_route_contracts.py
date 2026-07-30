from __future__ import annotations

import pytest

from soca.core.llm_tool_router import LLMToolRouter, _build_prompt
from soca.core.route_catalog import source_profile, validate_route_fields
from soca.core.runtime import AssistantRuntime
from soca.core.tool_routing import (
    ToolRouterConfig,
    build_evidence_completion_schema,
    build_route_decision_schema,
    parse_evidence_completion,
    parse_route_decision,
)
from soca.core.turn import RuntimeRoute
from soca.llm.base import LLMResult
from soca.tools import ToolRuntime
from tests.fake_tools import ReadOnlyInspectTool, ReadOnlySearchTool


def test_source_profiles_are_explicit_multilabel_contract() -> None:
    assert source_profile(()) == "neither"
    assert source_profile(("knowledge",)) == "knowledge"
    assert source_profile(("memory",)) == "memory"
    assert source_profile(("memory", "knowledge")) == "both"


def test_route_contract_rejects_handler_or_sources_on_wrong_disposition() -> None:
    with pytest.raises(ValueError, match="cannot name a handler"):
        validate_route_fields("smalltalk", handler="knowledge.inspect", sources=())
    with pytest.raises(ValueError, match="cannot select retrieval sources"):
        validate_route_fields("out_of_scope", handler=None, sources=("knowledge",))


def test_llm_route_parser_does_not_accept_legacy_tool_none_shape() -> None:
    parsed = parse_route_decision(
        '{"route":"retrieval_request","handler":null,"arguments":{},"sources":["knowledge","memory"]}',
        max_chars=512,
    )
    assert parsed.route == "retrieval_request"
    assert parsed.sources == ("knowledge", "memory")

    with pytest.raises(ValueError, match="invalid_root_fields"):
        parse_route_decision('{"tool":"none","arguments":{}}', max_chars=128)

    with pytest.raises(ValueError, match="retrieval_missing_sources"):
        parse_route_decision(
            '{"route":"retrieval_request","handler":null,"arguments":{},"sources":[]}',
            max_chars=512,
        )


def test_llm_router_prompt_exposes_inspect_as_navigation_not_evidence() -> None:
    prompt = _build_prompt(
        "Khái niệm kho tri thức nói chung là gì?",
        (
            {
                "name": "knowledge.inspect",
                "description": "Inspect the local vault structure.",
                "input_schema": {},
            },
        ),
    )

    assert "navigation metadata" in prompt
    assert "not a general-knowledge answer tool" in prompt
    assert "Classify the user's intent, not isolated words" in prompt
    assert "what the user wrote, noted, learned" in prompt
    assert "even when the subject also names a folder" in prompt


def test_route_schema_binds_each_handler_to_its_argument_contract() -> None:
    tools = ToolRuntime([ReadOnlyInspectTool(), ReadOnlySearchTool()])
    schema = build_route_decision_schema(tools.list_specs(include_disabled=False))
    search_branch = next(
        branch
        for branch in schema["oneOf"]
        if branch["properties"]["handler"].get("const") == "knowledge.search"
    )
    inspect_branch = next(
        branch
        for branch in schema["oneOf"]
        if branch["properties"]["handler"].get("const") == "knowledge.inspect"
    )

    assert search_branch["properties"]["arguments"]["required"] == ["query"]
    assert inspect_branch["properties"]["arguments"]["required"] == []
    assert search_branch["properties"]["sources"]["maxItems"] == 0


def test_evidence_completion_contract_binds_continuation_to_a_tool_schema() -> None:
    tools = ToolRuntime([ReadOnlySearchTool()])
    schema = build_evidence_completion_schema(tools.list_specs(include_disabled=False))
    search_branch = next(
        branch
        for branch in schema["oneOf"]
        if branch["properties"]["status"].get("const") == "continue"
    )

    assert search_branch["properties"]["arguments"]["required"] == ["query"]
    decision = parse_evidence_completion(
        '{"status":"continue","handler":"knowledge.search",'
        '"arguments":{"query":"attention","limit":3},'
        '"reason_code":"coverage_gap"}',
        max_chars=512,
    )
    assert decision.status == "continue"
    assert decision.call is not None
    assert decision.call.arguments["query"] == "attention"


class _FakeRouterLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, user_msg: str, **kwargs: object) -> LLMResult:
        del user_msg, kwargs
        return LLMResult(self.response, "", 0, 0, 0.0, 0.0, 0.0)


class _SequenceRouterLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, user_msg: str, **kwargs: object) -> LLMResult:
        del kwargs
        self.prompts.append(user_msg)
        return LLMResult(self.responses.pop(0), "", 0, 0, 0.0, 0.0, 0.0)


class _FailingRouterLLM:
    def generate(self, user_msg: str, **kwargs: object) -> LLMResult:
        del user_msg, kwargs
        raise RuntimeError("provider unavailable")


def test_llm_router_uses_route_contract_for_direct_and_retrieval() -> None:
    config = ToolRouterConfig(mode="llm", repair_attempts=0)
    direct = LLMToolRouter(
        _FakeRouterLLM(
            '{"route":"direct_tool","handler":"knowledge.inspect","arguments":{},"sources":[]}'
        ),
        ToolRuntime([ReadOnlyInspectTool()]),
        config=config,
    )
    call = direct.select("vault có gì", knowledge_limit=3)
    assert call is not None and call.name == "knowledge.inspect"
    assert direct.last_decision.disposition == "direct_tool"

    retrieval = LLMToolRouter(
        _FakeRouterLLM(
            '{"route":"retrieval_request","handler":null,"arguments":{},"sources":["memory"]}'
        ),
        ToolRuntime([ReadOnlyInspectTool()]),
        config=config,
    )
    assert retrieval.select("tôi đã chọn TTS gì", knowledge_limit=3) is None
    assert retrieval.last_decision.sources == ("memory",)
    assert retrieval.last_decision.source_profile == "memory"


def test_llm_unresolved_route_does_not_terminally_block_the_answer_model() -> None:
    router = LLMToolRouter(
        _FakeRouterLLM('{"route":"unresolved","handler":null,"arguments":{},"sources":[]}'),
        ToolRuntime([ReadOnlyInspectTool()]),
        config=ToolRouterConfig(mode="llm", repair_attempts=0),
    )
    result = AssistantRuntime(
        llm=_FakeRouterLLM("should not answer"),
        tool_runtime=ToolRuntime([ReadOnlyInspectTool()]),
        tool_router=router,
    ).run_text_turn("cái đó thế nào rồi?")

    assert result.route == RuntimeRoute.FREE_CHAT
    assert result.trace is not None
    assert result.trace.disposition == "unresolved"


def test_llm_invalid_output_fails_closed_without_deterministic_fallback() -> None:
    router = LLMToolRouter(
        _FakeRouterLLM("not json"),
        ToolRuntime([ReadOnlyInspectTool()]),
        config=ToolRouterConfig(mode="llm", repair_attempts=0),
    )

    assert router.select("vault có gì", knowledge_limit=3) is None
    assert router.last_tier == "llm"
    assert router.last_decision.reason == "llm_invalid_output:no_json_object"
    assert router.last_decision.disposition == "unresolved"


def test_json_schema_router_fails_closed_when_engine_lacks_structured_output() -> None:
    router = LLMToolRouter(
        _FakeRouterLLM(
            '{"route":"direct_tool","handler":"knowledge.inspect","arguments":{},"sources":[]}'
        ),
        ToolRuntime([ReadOnlyInspectTool()]),
        config=ToolRouterConfig(
            mode="llm",
            response_mode="json_schema",
            repair_attempts=0,
        ),
    )

    assert router.select("vault có gì", knowledge_limit=3) is None
    assert router.last_decision.reason == "llm_invalid_output:structured_output_unsupported"
    assert router.last_decision.disposition == "unresolved"


def test_llm_provider_failure_fails_closed_with_observable_reason() -> None:
    router = LLMToolRouter(
        _FailingRouterLLM(),
        ToolRuntime([ReadOnlyInspectTool()]),
        config=ToolRouterConfig(mode="llm", repair_attempts=0),
    )

    assert router.select("vault có gì", knowledge_limit=3) is None
    assert router.last_decision.reason == "llm_provider_failed"
    assert router.last_decision.disposition == "unresolved"


def test_llm_router_receives_manifest_context_and_can_refine_once() -> None:
    llm = _SequenceRouterLLM(
        [
            '{"route":"unresolved","handler":null,"arguments":{},"sources":[]}',
            '{"route":"direct_tool","handler":"knowledge.search",'
            '"arguments":{"query":"transformer","limit":3},"sources":[]}',
        ]
    )
    router = LLMToolRouter(
        llm,
        ToolRuntime([ReadOnlySearchTool()]),
        config=ToolRouterConfig(mode="llm", repair_attempts=0),
        vault_manifest_provider=lambda: (
            '{"tree":{"wiki/learning":["wiki/learning/transformer.md"]}}'
        ),
    )
    router.set_context(turn_context="Active goal: identify a note")

    assert router.select("ghi chú của tôi về transformer", knowledge_limit=3) is None
    assert '"tree"' in llm.prompts[0]
    assert "Active goal" in llm.prompts[0]

    refined = router.refine(
        "ghi chú của tôi về transformer",
        observation="retrieval evidence insufficient",
        knowledge_limit=3,
    )

    assert refined is not None
    assert refined.name == "knowledge.search"
    assert refined.arguments == {"query": "transformer", "limit": 3}
    assert llm.prompts[1].startswith("You are SoCa's bounded retrieval refiner.")
    assert "Original user request:" in llm.prompts[1]
    assert "User text:" not in llm.prompts[1]


def test_llm_router_assesses_goal_coverage_before_finalizing() -> None:
    llm = _SequenceRouterLLM(
        [
            '{"status":"continue","handler":"knowledge.search",'
            '"arguments":{"query":"unfinished work","limit":3},'
            '"reason_code":"partial_scope"}'
        ]
    )
    router = LLMToolRouter(
        llm,
        ToolRuntime([ReadOnlySearchTool()]),
        config=ToolRouterConfig(mode="llm", repair_attempts=0),
    )

    decision = router.assess_evidence(
        "check every unfinished task",
        observation='{"receipt":{"hits":[{"path":"wiki/weekly.md"}]}}',
        knowledge_limit=3,
    )

    assert decision.status == "continue"
    assert decision.call is not None
    assert decision.call.name == "knowledge.search"
    assert llm.prompts[0].startswith("You are SoCa's evidence-completion controller.")
    assert "covers every requested aspect" in llm.prompts[0]


def test_llm_router_logs_manifest_provider_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_manifest() -> str:
        raise OSError("catalog unavailable")

    router = LLMToolRouter(
        _FakeRouterLLM(
            '{"route":"smalltalk","handler":null,"arguments":{},"sources":[]}'
        ),
        ToolRuntime([ReadOnlyInspectTool()]),
        config=ToolRouterConfig(mode="llm", repair_attempts=0),
        vault_manifest_provider=fail_manifest,
    )

    with caplog.at_level("WARNING"):
        assert router.select("xin chào", knowledge_limit=3) is None

    assert "Vault manifest unavailable for capability routing (OSError)" in caplog.text
