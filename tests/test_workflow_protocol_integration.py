from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

from soca.core import AssistantRuntime, RuntimeOptions
from soca.core.workflow import EventType, TerminalStatus
from soca.core.workflow.protocol import (
    workflow_event_from_protocol,
    workflow_event_to_protocol,
)
from soca.tools import LocalTimeTool, ToolCall, ToolRuntime


def test_public_runtime_emits_serializable_protocol_v2_trajectory() -> None:
    runtime = AssistantRuntime(
        tool_runtime=ToolRuntime(
            [
                LocalTimeTool(
                    now_fn=lambda: datetime(
                        2026,
                        7,
                        30,
                        9,
                        30,
                        tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"),
                    )
                )
            ]
        ),
        options=RuntimeOptions(turn_workflow="shadow"),
    )

    result = runtime.run_controlled_workflow(
        "Cho tôi biết giờ local",
        explicit_call=ToolCall("local_time.now", {}),
        source="voice",
    )
    wire_events = [
        json.loads(json.dumps(workflow_event_to_protocol(event)))
        for event in result.events
    ]
    decoded = [workflow_event_from_protocol(event) for event in wire_events]

    assert result.terminal.status is TerminalStatus.ACHIEVED
    assert decoded == list(result.events)
    assert all(event.protocol_version == 2 for event in decoded)
    assert all(event.surface == "voice" for event in decoded)
    assert [event.sequence for event in decoded] == list(range(len(decoded)))
    assert sum(event.event is EventType.TURN_TERMINAL for event in decoded) == 1
    assert decoded[-1].event is EventType.TURN_TERMINAL
