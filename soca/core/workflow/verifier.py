from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from soca.tools import ToolResult

from .contracts import GoalContract


@dataclass(frozen=True)
class Verification:
    achieved: bool
    reason: str


VerifierRule = Callable[[GoalContract, ToolResult], Verification]


def verify_tool_result(goal: GoalContract, result: ToolResult) -> Verification:
    """Conservative default verifier; domain rules can be injected later."""
    del goal
    if not result.ok:
        return Verification(False, result.error or "tool_failed")
    if not result.content.strip() and not result.data:
        return Verification(False, "tool_returned_no_observation")
    return Verification(True, "tool_observation_available")


class DeterministicVerifier:
    def __init__(self, rules: dict[str, VerifierRule] | None = None) -> None:
        self.rules = dict(rules or {})

    def verify(self, goal: GoalContract, result: ToolResult) -> Verification:
        rule = self.rules.get(result.name, verify_tool_result)
        return rule(goal, result)
