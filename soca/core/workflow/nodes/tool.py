from __future__ import annotations

from dataclasses import dataclass

from soca.tools import ToolCall, ToolResult, ToolRuntime


@dataclass
class ToolExecutionNode:
    runtime: ToolRuntime

    def execute(self, call: ToolCall) -> ToolResult:
        return self.runtime.call(call)
