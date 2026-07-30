from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from soca.tools import ToolExecutionStatus, ToolResult

from .contracts import Capability, GoalContract, PlannedAction, SourceKind


@dataclass(frozen=True)
class Verification:
    achieved: bool
    reason: str
    unmet_criteria: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()


VerifierRule = Callable[[GoalContract, ToolResult], Verification]


_CAPABILITY_SOURCES = {
    Capability.KNOWLEDGE_CATALOG: SourceKind.KNOWLEDGE,
    Capability.KNOWLEDGE_SEARCH: SourceKind.KNOWLEDGE,
    Capability.KNOWLEDGE_READ: SourceKind.KNOWLEDGE,
    Capability.MEMORY_SEARCH: SourceKind.MEMORY,
}


def source_for_capability(capability: Capability) -> SourceKind | None:
    return _CAPABILITY_SOURCES.get(capability)


_CRITERION_SOURCES = {
    "knowledge_queried": SourceKind.KNOWLEDGE,
    "memory_queried": SourceKind.MEMORY,
}


def unmet_goal_criteria(
    goal: GoalContract,
    *,
    achieved_sources: set[SourceKind],
    has_observation: bool,
) -> tuple[str, ...]:
    unmet: list[str] = []
    for criterion in goal.success_criteria:
        required_source = _CRITERION_SOURCES.get(criterion.kind)
        if required_source is not None:
            if required_source not in achieved_sources:
                unmet.append(criterion.kind)
            continue
        if criterion.kind == "tool_observation_available":
            if not has_observation:
                unmet.append(criterion.kind)
            continue
        unmet.append(criterion.kind)
    return tuple(unmet)


def verify_tool_result(
    goal: GoalContract,
    result: ToolResult,
    action: PlannedAction | None = None,
) -> Verification:
    if not result.ok:
        return Verification(False, tool_error_code(result))
    if action is not None and goal.required_sources:
        action_source = source_for_capability(action.capability)
        if action_source not in goal.required_sources:
            return Verification(
                False,
                "required_source_not_used",
                tuple(source.value for source in goal.required_sources),
            )
    hits = result.data.get("hits")
    if isinstance(hits, list) and not hits:
        return Verification(False, "no_matching_observation")
    if not result.content.strip() and not result.data:
        return Verification(False, "tool_returned_no_observation")
    evidence_ids: list[str] = []
    if isinstance(hits, list):
        evidence_ids.extend(
            str(item.get("path")) for item in hits if isinstance(item, dict) and item.get("path")
        )
    path = result.data.get("path")
    if isinstance(path, str) and path:
        evidence_ids.append(path)
    if action is not None and action.expected_observation and not result.content.strip():
        return Verification(False, "expected_observation_missing")
    return Verification(
        True,
        "tool_observation_available",
        evidence_ids=tuple(dict.fromkeys(evidence_ids)),
    )


_STATUS_FAILURE_CODES = {
    ToolExecutionStatus.NOT_FOUND: "tool_not_found",
    ToolExecutionStatus.INVALID: "invalid_tool_input",
    ToolExecutionStatus.DENIED: "tool_denied",
    ToolExecutionStatus.TRANSIENT_ERROR: "tool_transient_error",
    ToolExecutionStatus.PERMANENT_ERROR: "tool_failed",
    ToolExecutionStatus.CANCELLED: "cancelled",
}


def tool_error_code(result: ToolResult) -> str:
    """Return stable telemetry without exposing exception text."""
    raw = result.error.strip().lower()
    if raw and raw.replace("_", "").isalnum() and all(
        character.isalnum() or character == "_" for character in raw
    ):
        return raw
    status = result.status
    if status is not None:
        return _STATUS_FAILURE_CODES[status]
    return "tool_failed"


class DeterministicVerifier:
    def __init__(self, rules: dict[str, VerifierRule] | None = None) -> None:
        self.rules = dict(rules or {})

    def verify(
        self,
        goal: GoalContract,
        result: ToolResult,
        action: PlannedAction | None = None,
    ) -> Verification:
        rule = self.rules.get(result.name, verify_tool_result)
        if rule is verify_tool_result:
            return verify_tool_result(goal, result, action)
        return rule(goal, result)
