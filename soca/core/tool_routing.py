from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from soca.config import DEFAULT_MAX_TOKENS
from soca.tools import ToolCall, ToolSpec

SourceProfile = Literal["knowledge", "memory", "both", "neither"]

ToolRouterMode = Literal["deterministic", "llm", "cascade"]
RouterResponseMode = Literal["prompt_json", "json_schema"]
TurnDisposition = Literal[
    "direct_tool",
    "retrieval_request",
    "smalltalk",
    "out_of_scope",
    "unresolved",
]
EvidenceCompletionStatus = Literal["complete", "continue", "insufficient"]


@dataclass(frozen=True)
class SemanticRouterConfig:
    enabled: bool = False
    threshold: float = 0.0
    margin: float = 0.0
    direct_tool_threshold: float = 0.85
    direct_tool_retrieval_margin: float = 0.01
    examples_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("semantic enabled must be a boolean")
        for name, value in (
            ("threshold", self.threshold),
            ("margin", self.margin),
            ("direct_tool_threshold", self.direct_tool_threshold),
            ("direct_tool_retrieval_margin", self.direct_tool_retrieval_margin),
        ):
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
    max_tokens: int = DEFAULT_MAX_TOKENS
    max_output_chars: int = 8_192
    repair_attempts: int = 1
    zero_data_retention: bool = True
    enabled_in_voice: bool = True
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
        ):
            if not isinstance(value, bool):
                raise ValueError("router flags must be booleans")
        if not isinstance(self.semantic, SemanticRouterConfig):
            raise ValueError("semantic router config is invalid")


@dataclass(frozen=True)
class ParsedToolDecision:
    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class EvidenceCompletionDecision:
    status: EvidenceCompletionStatus
    call: ToolCall | None = None
    reason_code: str = ""


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
    handler: str | None = None
    selected_routes: tuple[TurnDisposition, ...] = ()
    sources: tuple[str, ...] = ()
    source_profile: SourceProfile | None = None
    scores: dict[str, float] = field(default_factory=dict)
    source_scores: dict[str, float] = field(default_factory=dict)
    runner_up: str | None = None
    margin: float | None = None


@dataclass(frozen=True)
class ParsedRouteDecision:
    route: TurnDisposition
    handler: str | None
    arguments: dict[str, Any]
    sources: tuple[str, ...]


def _parse_single_json_object(raw: str, *, max_chars: int) -> dict[str, Any]:
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
    if len(objects) != 1:
        raise RouterOutputError("no_json_object" if not objects else "multiple_json_objects")
    return objects[0]


def build_evidence_completion_schema(specs: tuple[ToolSpec, ...]) -> dict[str, Any]:
    retrieval_specs = tuple(
        spec
        for spec in specs
        if spec.name in {"knowledge.inspect", "knowledge.search", "knowledge.read"}
    )
    return _flat_decision_schema(
        discriminator_name="status",
        discriminator_values=("complete", "continue", "insufficient"),
        specs=retrieval_specs,
        extra_properties={
            "reason_code": {"type": "string", "minLength": 1, "maxLength": 80},
        },
    )


def parse_evidence_completion(raw: str, *, max_chars: int) -> EvidenceCompletionDecision:
    payload = _parse_single_json_object(raw, max_chars=max_chars)
    if set(payload) != {"status", "handler", "arguments", "reason_code"}:
        raise RouterOutputError("invalid_completion_fields")
    status = payload["status"]
    handler = payload["handler"]
    arguments = _remove_null_arguments(payload["arguments"])
    reason_code = payload["reason_code"]
    if status not in {"complete", "continue", "insufficient"}:
        raise RouterOutputError("invalid_completion_status")
    if not isinstance(arguments, dict) or any(not isinstance(key, str) for key in arguments):
        raise RouterOutputError("completion_arguments_not_object")
    if not isinstance(reason_code, str) or not 1 <= len(reason_code) <= 80:
        raise RouterOutputError("invalid_completion_reason")
    if status == "continue":
        if not isinstance(handler, str) or not handler:
            raise RouterOutputError("invalid_completion_handler")
        return EvidenceCompletionDecision(
            status="continue",
            call=ToolCall(handler, dict(arguments)),
            reason_code=reason_code,
        )
    handler = None
    arguments = {}
    return EvidenceCompletionDecision(
        status=cast(EvidenceCompletionStatus, status),
        reason_code=reason_code,
    )


def build_route_decision_schema(specs: tuple[ToolSpec, ...]) -> dict[str, Any]:
    """Build a provider-portable strict schema for the route contract.

    OpenAI-compatible structured-output endpoints reject a root ``oneOf`` even
    though it is valid JSON Schema.  The discriminator is therefore explicit
    and the local parser remains responsible for cross-field invariants such as
    ``retrieval_request`` requiring sources and ``direct_tool`` requiring
    valid arguments for the selected handler.
    """
    return _flat_decision_schema(
        discriminator_name="route",
        discriminator_values=(
            "direct_tool",
            "retrieval_request",
            "smalltalk",
            "out_of_scope",
            "unresolved",
        ),
        specs=specs,
        extra_properties={
            "sources": {
                "type": "array",
                "items": {"type": "string", "enum": ["knowledge", "memory"]},
            },
        },
    )


def _flat_decision_schema(
    *,
    discriminator_name: str,
    discriminator_values: tuple[str, ...],
    specs: tuple[ToolSpec, ...],
    extra_properties: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Encode a discriminated contract without unsupported root unions."""
    argument_properties = _nullable_argument_properties(specs)
    argument_names = sorted(argument_properties)
    # Keep this nullable and validate the selected name against the live tool
    # runtime after parsing. Binding the enum here encourages some models to
    # emit a handler even when the discriminator says retrieval/smalltalk.
    # Cross-field dependencies are intentionally enforced by the parser.
    handler_schema: dict[str, Any] = {"type": ["string", "null"]}

    properties: dict[str, Any] = {
        discriminator_name: {"type": "string", "enum": list(discriminator_values)},
        "handler": handler_schema,
        "arguments": {
            "type": "object",
            "properties": argument_properties,
            "required": argument_names,
            "additionalProperties": False,
        },
    }
    properties.update(deepcopy(extra_properties))
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def build_nullable_arguments_schema(specs: tuple[ToolSpec, ...]) -> dict[str, Any]:
    """Build a strict nullable argument object for provider structured output."""
    properties = _nullable_argument_properties(specs)
    return {
        "type": "object",
        "properties": properties,
        "required": sorted(properties),
        "additionalProperties": False,
    }


def _nullable_argument_properties(specs: tuple[ToolSpec, ...]) -> dict[str, dict[str, Any]]:
    """Merge enabled tool argument fields into one strict nullable object."""
    properties: dict[str, dict[str, Any]] = {}
    for spec in specs:
        for name, schema in spec.input_schema.get("properties", {}).items():
            candidate = _nullable_schema(schema)
            existing = properties.get(name)
            if existing is None:
                properties[name] = candidate
            elif _schema_shape(existing) != _schema_shape(candidate):
                properties[name] = {"anyOf": [existing, candidate]}
    return properties


def _nullable_schema(schema: dict[str, Any]) -> dict[str, Any]:
    candidate = deepcopy(schema)
    schema_type = candidate.get("type")
    if isinstance(schema_type, str):
        candidate["type"] = [schema_type, "null"]
    elif isinstance(schema_type, list) and "null" not in schema_type:
        candidate["type"] = [*schema_type, "null"]
    else:
        candidate = {"anyOf": [candidate, {"type": "null"}]}
    return candidate


def _schema_shape(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _schema_shape(item)
            for key, item in value.items()
            if key not in {"description", "title"}
        }
    if isinstance(value, list):
        return [_schema_shape(item) for item in value]
    return value


def _remove_null_arguments(arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        return arguments
    return {key: value for key, value in arguments.items() if value is not None}


def parse_route_decision(raw: str, *, max_chars: int) -> ParsedRouteDecision:
    payload = _parse_single_json_object(raw, max_chars=max_chars)
    if set(payload) != {"route", "handler", "arguments", "sources"}:
        raise RouterOutputError("invalid_root_fields")
    route = payload["route"]
    handler = payload["handler"]
    arguments = _remove_null_arguments(payload["arguments"])
    sources = payload["sources"]
    if route not in {
        "direct_tool",
        "retrieval_request",
        "smalltalk",
        "out_of_scope",
        "unresolved",
    }:
        raise RouterOutputError("invalid_route")
    if handler is not None and (not isinstance(handler, str) or not handler):
        raise RouterOutputError("invalid_handler")
    if not isinstance(arguments, dict) or any(not isinstance(key, str) for key in arguments):
        raise RouterOutputError("arguments_not_object")
    if not isinstance(sources, list) or not all(
        source in {"knowledge", "memory"} for source in sources
    ):
        raise RouterOutputError("invalid_sources")
    if len(set(sources)) != len(sources):
        raise RouterOutputError("duplicate_sources")
    if route == "direct_tool" and handler is None:
        raise RouterOutputError("direct_route_missing_handler")
    if route == "retrieval_request" and not sources:
        raise RouterOutputError("retrieval_missing_sources")
    if route != "retrieval_request" and sources:
        raise RouterOutputError("non_retrieval_route_has_sources")
    if route != "direct_tool":
        # The provider schema cannot express the dependency between the route
        # discriminator and these fields without a root union. They carry no
        # executable meaning on non-direct routes, so normalize them away;
        # only a direct_tool decision reaches the tool validator.
        handler = None
        arguments = {}
    return ParsedRouteDecision(
        route=route,
        handler=handler,
        arguments=dict(arguments),
        sources=tuple(sources),
    )


def build_tool_decision_schema(specs: tuple[ToolSpec, ...]) -> dict[str, Any]:
    names = ["none", *sorted(spec.name for spec in specs)]
    return {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": names},
            "arguments": build_nullable_arguments_schema(specs),
        },
        "required": ["tool", "arguments"],
        "additionalProperties": False,
    }


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
    arguments = _remove_null_arguments(payload["arguments"])
    if not isinstance(tool, str) or not tool:
        raise RouterOutputError("invalid_tool_name")
    if not isinstance(arguments, dict) or any(not isinstance(key, str) for key in arguments):
        raise RouterOutputError("arguments_not_object")
    if tool == "none" and arguments:
        raise RouterOutputError("none_has_arguments")
    return ParsedToolDecision(tool=tool, arguments=dict(arguments))
