from __future__ import annotations

import json

import pytest

from soca.core.tool_routing import (
    RouterOutputError,
    SemanticRouterConfig,
    ToolRouterConfig,
    parse_tool_decision,
)


def test_parser_accepts_one_object_inside_prose() -> None:
    result = parse_tool_decision(
        'Result:\n```json\n{"tool":"none","arguments":{}}\n```',
        max_chars=200,
    )
    assert result.tool == "none"
    assert result.arguments == {}


def test_parser_rejects_multiple_top_level_objects() -> None:
    raw = json.dumps({"tool": "none", "arguments": {}}) + " " + json.dumps(
        {"tool": "none", "arguments": {}}
    )
    with pytest.raises(RouterOutputError, match="multiple_json_objects"):
        parse_tool_decision(raw, max_chars=200)


def test_parser_rejects_none_arguments_and_output_cap() -> None:
    with pytest.raises(RouterOutputError, match="none_has_arguments"):
        parse_tool_decision('{"tool":"none","arguments":{"x":1}}', max_chars=100)
    with pytest.raises(RouterOutputError, match="output_too_large"):
        parse_tool_decision("x" * 101, max_chars=100)


def test_router_config_rejects_bool_numeric_and_missing_examples() -> None:
    with pytest.raises(ValueError):
        ToolRouterConfig(max_tokens=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SemanticRouterConfig(enabled=True)
