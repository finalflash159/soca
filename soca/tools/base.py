from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class SideEffectLevel(Enum):
    READ_ONLY = "read_only"
    LOCAL_STATE = "local_state"
    NETWORK = "network"
    DEVICE = "device"


SIDE_EFFECT_RANK = {
    SideEffectLevel.READ_ONLY: 0,
    SideEffectLevel.LOCAL_STATE: 1,
    SideEffectLevel.NETWORK: 2,
    SideEffectLevel.DEVICE: 3,
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    side_effect: SideEffectLevel = SideEffectLevel.READ_ONLY
    enabled: bool = True


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    name: str
    ok: bool
    content: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class Tool(Protocol):
    @property
    def spec(self) -> ToolSpec:
        ...

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        ...


def object_schema(
    properties: dict[str, dict[str, Any]] | None = None,
    required: list[str] | None = None,
    additional_properties: bool = False,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties or {},
        "required": required or [],
        "additionalProperties": additional_properties,
    }


def _validate_tool_spec(spec: ToolSpec) -> None:
    if not spec.name.strip():
        raise ValueError("Tool name must not be empty")
    if not spec.description.strip():
        raise ValueError(f"Tool {spec.name} description must not be empty")
    if spec.input_schema.get("type") != "object":
        raise ValueError(f"Tool {spec.name} input_schema must be an object schema")
    if not isinstance(spec.input_schema.get("properties", {}), dict):
        raise ValueError(f"Tool {spec.name} schema properties must be a dict")
    required = spec.input_schema.get("required", [])
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        raise ValueError(f"Tool {spec.name} schema required must be a list of strings")


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _matches_json_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (isinstance(value, int | float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return True


def validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> str:
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    allow_extra = bool(schema.get("additionalProperties", True))

    for key in required:
        if key not in arguments:
            return f"Missing required argument: {key}"

    if not allow_extra:
        extra_keys = sorted(set(arguments) - set(properties))
        if extra_keys:
            return f"Unexpected argument(s): {', '.join(extra_keys)}"

    for key, value in arguments.items():
        property_schema = properties.get(key)
        if property_schema is None:
            continue
        expected_type = property_schema.get("type")
        if isinstance(expected_type, str) and not _matches_json_type(value, expected_type):
            return (
                f"Argument {key} must be {expected_type}; "
                f"got {_type_name(value)}"
            )

    return ""


class ToolRuntime:
    def __init__(
        self,
        tools: Iterable[Tool] = (),
        max_side_effect: SideEffectLevel = SideEffectLevel.LOCAL_STATE,
    ) -> None:
        self.max_side_effect = max_side_effect
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        spec = tool.spec
        _validate_tool_spec(spec)
        if spec.name in self._tools:
            raise ValueError(f"Duplicate tool name: {spec.name}")
        self._tools[spec.name] = tool

    def list_specs(self, include_disabled: bool = True) -> tuple[ToolSpec, ...]:
        specs = [tool.spec for tool in self._tools.values()]
        if not include_disabled:
            specs = [spec for spec in specs if spec.enabled]
        return tuple(sorted(specs, key=lambda spec: spec.name))

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def call(self, call: ToolCall) -> ToolResult:
        tool = self.get(call.name)
        if tool is None:
            return ToolResult(
                name=call.name,
                ok=False,
                content="",
                error=f"Unknown tool: {call.name}",
            )

        spec = tool.spec
        if not spec.enabled:
            return ToolResult(
                name=spec.name,
                ok=False,
                content="",
                error=f"Tool is disabled: {spec.name}",
            )

        if SIDE_EFFECT_RANK[spec.side_effect] > SIDE_EFFECT_RANK[self.max_side_effect]:
            return ToolResult(
                name=spec.name,
                ok=False,
                content="",
                error=(
                    f"Tool side effect {spec.side_effect.value} exceeds "
                    f"allowed level {self.max_side_effect.value}"
                ),
            )

        validation_error = validate_arguments(spec.input_schema, call.arguments)
        if validation_error:
            return ToolResult(
                name=spec.name,
                ok=False,
                content="",
                error=validation_error,
            )

        try:
            return tool.run(call.arguments)
        except Exception as exc:
            return ToolResult(
                name=spec.name,
                ok=False,
                content="",
                error=str(exc),
            )
