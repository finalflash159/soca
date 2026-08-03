from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, cast

from .contracts import TurnNode, TurnSource
from .events import (
    PROTOCOL_VERSION,
    EventStatus,
    EventType,
    WorkflowEvent,
)

CURRENT_PROTOCOL_VERSION = PROTOCOL_VERSION
SUPPORTED_PROTOCOL_VERSIONS = (CURRENT_PROTOCOL_VERSION,)


def protocol_hello(*, profile: str, no_model: bool, stack: dict[str, Any]) -> dict[str, Any]:
    return {
        "event": "hello",
        "version": CURRENT_PROTOCOL_VERSION,
        "protocol_version": CURRENT_PROTOCOL_VERSION,
        "supported_versions": list(SUPPORTED_PROTOCOL_VERSIONS),
        "profile": profile,
        "no_model": no_model,
        "stack": stack,
    }


def workflow_event_to_protocol(event: WorkflowEvent) -> dict[str, Any]:
    return {
        "event": event.event.value,
        "protocol_version": event.protocol_version,
        "session_id": event.session_id,
        "run_id": event.run_id,
        "goal_id": event.goal_id,
        "sequence": event.sequence,
        "surface": event.surface,
        "timestamp": event.timestamp,
        "node": event.node.value,
        "status": event.status.value,
        "payload": _json_value(event.payload),
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def workflow_event_from_protocol(payload: dict[str, Any]) -> WorkflowEvent:
    required_strings = {
        "session_id",
        "run_id",
        "goal_id",
        "surface",
        "timestamp",
        "node",
        "status",
        "event",
    }
    if payload.get("protocol_version") != CURRENT_PROTOCOL_VERSION:
        raise ValueError("unsupported workflow protocol version")
    if any(not isinstance(payload.get(key), str) or not payload[key] for key in required_strings):
        raise ValueError("workflow event envelope contains an invalid string field")
    sequence = payload.get("sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("workflow event sequence must be a non-negative integer")
    event_payload = payload.get("payload")
    if not isinstance(event_payload, dict):
        raise ValueError("workflow event payload must be an object")
    surface = payload["surface"]
    if surface not in {"ask", "cli", "chat", "voice"}:
        raise ValueError("workflow event surface is invalid")
    try:
        datetime.fromisoformat(payload["timestamp"].replace("Z", "+00:00"))
        event = EventType(payload["event"])
        node = TurnNode(payload["node"])
        status = EventStatus(payload["status"])
    except (ValueError, TypeError) as exc:
        raise ValueError("workflow event envelope contains an invalid enum or timestamp") from exc
    return WorkflowEvent(
        event=event,
        session_id=payload["session_id"],
        run_id=payload["run_id"],
        goal_id=payload["goal_id"],
        sequence=sequence,
        surface=cast(TurnSource, surface),
        timestamp=payload["timestamp"],
        node=node,
        status=status,
        payload=event_payload,
    )
