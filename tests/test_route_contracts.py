from __future__ import annotations

import pytest

from soca.core.llm_tool_router import LLMToolRouter
from soca.core.route_catalog import source_profile, validate_route_fields
from soca.core.runtime import AssistantRuntime
from soca.core.tool_routing import ToolRouterConfig, parse_route_decision
from soca.core.turn import RuntimeRoute
from soca.llm.base import LLMResult
from soca.tools import LocalTimeTool, ToolRuntime


def test_source_profiles_are_explicit_multilabel_contract() -> None:
    assert source_profile(()) == "neither"
    assert source_profile(("knowledge",)) == "knowledge"
    assert source_profile(("memory",)) == "memory"
    assert source_profile(("memory", "knowledge")) == "both"


def test_route_contract_rejects_handler_or_sources_on_wrong_disposition() -> None:
    with pytest.raises(ValueError, match="cannot name a handler"):
        validate_route_fields("smalltalk", handler="local_time.now", sources=())
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


class _FakeRouterLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, user_msg: str, **kwargs: object) -> LLMResult:
        del user_msg, kwargs
        return LLMResult(self.response, "", 0, 0, 0.0, 0.0, 0.0)


class _FailingRouterLLM:
    def generate(self, user_msg: str, **kwargs: object) -> LLMResult:
        del user_msg, kwargs
        raise RuntimeError("provider unavailable")


def test_llm_router_uses_route_contract_for_direct_and_retrieval() -> None:
    config = ToolRouterConfig(mode="llm", repair_attempts=0)
    direct = LLMToolRouter(
        _FakeRouterLLM(
            '{"route":"direct_tool","handler":"local_time.now","arguments":{},"sources":[]}'
        ),
        ToolRuntime([LocalTimeTool()]),
        config=config,
    )
    call = direct.select("mấy giờ rồi", knowledge_limit=3)
    assert call is not None and call.name == "local_time.now"
    assert direct.last_decision.disposition == "direct_tool"

    retrieval = LLMToolRouter(
        _FakeRouterLLM(
            '{"route":"retrieval_request","handler":null,"arguments":{},"sources":["memory"]}'
        ),
        ToolRuntime([LocalTimeTool()]),
        config=config,
    )
    assert retrieval.select("tôi đã chọn TTS gì", knowledge_limit=3) is None
    assert retrieval.last_decision.sources == ("memory",)
    assert retrieval.last_decision.source_profile == "memory"


def test_llm_unresolved_route_becomes_clarification_not_free_chat() -> None:
    router = LLMToolRouter(
        _FakeRouterLLM(
            '{"route":"unresolved","handler":null,"arguments":{},"sources":[]}'
        ),
        ToolRuntime([LocalTimeTool()]),
        config=ToolRouterConfig(mode="llm", repair_attempts=0),
    )
    result = AssistantRuntime(
        llm=_FakeRouterLLM("should not answer"),
        tool_runtime=ToolRuntime([LocalTimeTool()]),
        tool_router=router,
    ).run_text_turn("cái đó thế nào rồi?")

    assert result.route == RuntimeRoute.CLARIFICATION
    assert result.trace is not None
    assert result.trace.disposition == "unresolved"


def test_llm_invalid_output_fails_closed_without_deterministic_fallback() -> None:
    router = LLMToolRouter(
        _FakeRouterLLM("not json"),
        ToolRuntime([LocalTimeTool()]),
        config=ToolRouterConfig(mode="llm", repair_attempts=0),
    )

    assert router.select("mấy giờ rồi", knowledge_limit=3) is None
    assert router.last_tier == "llm"
    assert router.last_decision.reason == "llm_invalid_output:no_json_object"
    assert router.last_decision.disposition == "unresolved"


def test_llm_provider_failure_fails_closed_with_observable_reason() -> None:
    router = LLMToolRouter(
        _FailingRouterLLM(),
        ToolRuntime([LocalTimeTool()]),
        config=ToolRouterConfig(mode="llm", repair_attempts=0),
    )

    assert router.select("mấy giờ rồi", knowledge_limit=3) is None
    assert router.last_decision.reason == "llm_provider_failed"
    assert router.last_decision.disposition == "unresolved"
