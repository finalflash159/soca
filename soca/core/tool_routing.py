from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from soca.tools import ToolCall, ToolSpec

ToolRouterMode = Literal["deterministic", "llm", "cascade"]
RouterResponseMode = Literal["prompt_json", "json_schema"]
TurnDisposition = Literal[
    "direct_tool",
    "retrieval_request",
    "smalltalk",
    "out_of_scope",
    "unresolved",
]


@dataclass(frozen=True)
class SemanticRouterConfig:
    enabled: bool = False
    threshold: float = 0.0
    margin: float = 0.0
    examples_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("semantic enabled must be a boolean")
        for name, value in (("threshold", self.threshold), ("margin", self.margin)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"semantic {name} must be numeric")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"semantic {name} must be in [0, 1]")
        if self.enabled and self.examples_path is None:
            raise ValueError("semantic examples path is required when enabled")


@dataclass(frozen=True)
class ToolRouterConfig:
    mode: ToolRouterMode = "deterministic"
    response_mode: RouterResponseMode = "prompt_json"
    max_tokens: int = 96
    max_output_chars: int = 8_192
    repair_attempts: int = 1
    zero_data_retention: bool = True
    enabled_in_voice: bool = False
    semantic_enabled_in_voice: bool = False
    semantic: SemanticRouterConfig = field(default_factory=SemanticRouterConfig)

    def __post_init__(self) -> None:
        if self.mode not in {"deterministic", "llm", "cascade"}:
            raise ValueError("unknown tool router mode")
        if self.response_mode not in {"prompt_json", "json_schema"}:
            raise ValueError("unknown router response mode")
        values = (self.max_tokens, self.max_output_chars, self.repair_attempts)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError("router numeric config must contain integers")
        if self.max_tokens < 1 or self.max_output_chars < 64:
            raise ValueError("router token/output limits are invalid")
        if self.repair_attempts not in {0, 1}:
            raise ValueError("router supports zero or one repair attempt")
        for value in (
            self.zero_data_retention,
            self.enabled_in_voice,
            self.semantic_enabled_in_voice,
        ):
            if not isinstance(value, bool):
                raise ValueError("router flags must be booleans")
        if not isinstance(self.semantic, SemanticRouterConfig):
            raise ValueError("semantic router config is invalid")


@dataclass(frozen=True)
class ParsedToolDecision:
    tool: str
    arguments: dict[str, Any]


class RouterOutputError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ToolRouterDecision:
    """Observable capability decision, independent from executable tools.

    A ``call`` is intentionally populated only for a direct, allow-listed
    capability.  Retrieval is a disposition plus a source set; it is resolved
    by ``AssistantRuntime`` so a source can be ``both`` without inventing a
    fake executable tool.
    """

    call: ToolCall | None = None
    reason: str = "no_match"
    disposition: TurnDisposition = "unresolved"
    sources: tuple[str, ...] = ()
    scores: dict[str, float] = field(default_factory=dict)
    runner_up: str | None = None
    margin: float | None = None


def build_tool_decision_schema(specs: tuple[ToolSpec, ...]) -> dict[str, Any]:
    branches: list[dict[str, Any]] = [
        {
            "type": "object",
            "properties": {
                "tool": {"const": "none"},
                "arguments": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
            "required": ["tool", "arguments"],
            "additionalProperties": False,
        }
    ]
    for spec in specs:
        branches.append(
            {
                "type": "object",
                "properties": {
                    "tool": {"const": spec.name},
                    "arguments": dict(spec.input_schema),
                },
                "required": ["tool", "arguments"],
                "additionalProperties": False,
            }
        )
    return {"oneOf": branches}


def parse_tool_decision(raw: str, *, max_chars: int) -> ParsedToolDecision:
    if not isinstance(raw, str):
        raise RouterOutputError("output_not_string")
    if max_chars < 1 or len(raw) > max_chars:
        raise RouterOutputError("output_too_large")

    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(raw):
        offset = raw.find("{", cursor)
        if offset < 0:
            break
        try:
            candidate, consumed = decoder.raw_decode(raw[offset:])
        except json.JSONDecodeError:
            cursor = offset + 1
            continue
        cursor = offset + max(consumed, 1)
        if isinstance(candidate, dict):
            objects.append(candidate)

    if len(objects) == 0:
        raise RouterOutputError("no_json_object")
    if len(objects) != 1:
        raise RouterOutputError("multiple_json_objects")

    payload = objects[0]
    if set(payload) != {"tool", "arguments"}:
        raise RouterOutputError("invalid_root_fields")
    tool = payload["tool"]
    arguments = payload["arguments"]
    if not isinstance(tool, str) or not tool:
        raise RouterOutputError("invalid_tool_name")
    if not isinstance(arguments, dict) or any(not isinstance(key, str) for key in arguments):
        raise RouterOutputError("arguments_not_object")
    if tool == "none" and arguments:
        raise RouterOutputError("none_has_arguments")
    return ParsedToolDecision(tool=tool, arguments=dict(arguments))
