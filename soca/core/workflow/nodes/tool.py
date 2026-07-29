from __future__ import annotations

from dataclasses import dataclass

from soca.tools import ToolCall, ToolRuntime

from ..contracts import NodeOutcome, NodeStatus, TurnState


@dataclass
class ToolExecutionNode:
    runtime: ToolRuntime

    def execute(self, call: ToolCall) -> NodeOutcome:
        result = self.runtime.call(call)
        return NodeOutcome(
            node="execute_tool",
            status=NodeStatus.CONTINUE if result.ok else NodeStatus.RETRY,
            state=TurnState.EXECUTING,
            output=result,
            reason=result.error,
            retryable=not result.ok,
        )
