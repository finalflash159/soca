from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from soca.tools.base import ToolExecutionStatus

if TYPE_CHECKING:
    from .events import WorkflowEvent

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = (
    JSONScalar
    | list["JSONValue"]
    | tuple["JSONValue", ...]
    | dict[str, "JSONValue"]
    | Mapping[str, "JSONValue"]
)
TurnSource: TypeAlias = Literal["ask", "cli", "chat", "voice"]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return _freeze_json(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, StrEnum):
        return value.value
    return value


def _list_value(value: Any, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"checkpoint {field_name} must be a list")
    return value


def _mapping_list(value: Any, field_name: str) -> list[Mapping[str, Any]]:
    values = _list_value(value, field_name)
    if not all(isinstance(item, Mapping) for item in values):
        raise ValueError(f"checkpoint {field_name} items must be objects")
    return values


class GoalStatus(StrEnum):
    NEW = "new"
    ACTIVE = "active"
    WAITING_FOR_USER = "waiting_for_user"
    ACHIEVED = "achieved"
    ABANDONED = "abandoned"
    SUPERSEDED = "superseded"
    FAILED = "failed"


class SourceKind(StrEnum):
    KNOWLEDGE = "knowledge"
    MEMORY = "memory"
    LOCAL_TIME = "local_time"


class Capability(StrEnum):
    FREE_CHAT = "free_chat"
    KNOWLEDGE_SEARCH = "knowledge_search"
    KNOWLEDGE_READ = "knowledge_read"
    MEMORY_SEARCH = "memory_search"
    MEMORY_PROPOSE_NOTE = "memory_propose_note"
    LOCAL_TIME = "local_time"
    CLARIFICATION = "clarification"
    OUT_OF_SCOPE = "out_of_scope"


class TurnNode(StrEnum):
    ADMIT = "admit"
    RESOLVE_GOAL = "resolve_goal"
    CHOOSE_CAPABILITY = "choose_capability"
    MAKE_PLAN = "make_plan"
    AUTHORIZE_ACTION = "authorize_action"
    EXECUTE_ACTION = "execute_action"
    ASSESS_OBSERVATION = "assess_observation"
    REVISE_QUERY = "revise_query"
    SYNTHESIZE = "synthesize"
    VERIFY_ANSWER = "verify_answer"
    REPAIR_ANSWER = "repair_answer"
    ASK_CLARIFICATION = "ask_clarification"
    FINALIZE = "finalize"


class SideEffectClass(StrEnum):
    NONE = "none"
    READ_ONLY = "read_only"
    LOCAL_STATE = "local_state"
    EXTERNAL = "external"


class TerminalStatus(StrEnum):
    ACHIEVED = "achieved"
    NEEDS_CLARIFICATION = "needs_clarification"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    SAFE_FAILURE = "safe_failure"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CANCELLED = "cancelled"
    SYSTEM_FAILURE = "system_failure"


class EvidenceStatus(StrEnum):
    NOT_REQUESTED = "not_requested"
    SUPPORTED = "supported"
    WEAK = "weak"
    CONFLICTING = "conflicting"
    INSUFFICIENT = "insufficient"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class SuccessCriterion:
    kind: str
    description: str = ""
    source: SourceKind | None = None

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("success criterion kind must not be empty")


@dataclass(frozen=True)
class GoalConstraint:
    kind: str
    value: JSONValue

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("goal constraint kind must not be empty")
        object.__setattr__(self, "value", _freeze_json(self.value))


@dataclass(frozen=True)
class ResolvedEntity:
    surface: str
    canonical: str
    confidence: float

    def __post_init__(self) -> None:
        if not self.surface.strip() or not self.canonical.strip():
            raise ValueError("resolved entity text must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("resolved entity confidence must be between zero and one")


@dataclass(frozen=True)
class UnresolvedEntity:
    surface: str
    reason: str

    def __post_init__(self) -> None:
        if not self.surface.strip() or not self.reason.strip():
            raise ValueError("unresolved entity fields must not be empty")


@dataclass(frozen=True)
class GoalContract:
    goal_id: str
    objective: str
    success_criteria: tuple[SuccessCriterion, ...] = ()
    constraints: tuple[GoalConstraint, ...] = ()
    required_sources: tuple[SourceKind, ...] = ()
    resolved_entities: tuple[ResolvedEntity, ...] = ()
    unresolved_entities: tuple[UnresolvedEntity, ...] = ()
    status: GoalStatus = GoalStatus.ACTIVE
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    parent_goal_id: str | None = None

    def __post_init__(self) -> None:
        goal_id = self.goal_id.strip()
        objective = self.objective.strip()
        if not goal_id:
            raise ValueError("goal id must not be empty")
        if not objective:
            raise ValueError("goal objective must not be empty")
        if len(set(self.required_sources)) != len(self.required_sources):
            raise ValueError("required goal sources must be unique")
        object.__setattr__(self, "goal_id", goal_id)
        object.__setattr__(self, "objective", objective)

    def planner_text(self) -> str:
        parts = [f"Objective: {self.objective}"]
        if self.constraints:
            parts.append(
                "Constraints: "
                + ", ".join(f"{item.kind}={item.value}" for item in self.constraints)
            )
        if self.required_sources:
            parts.append(
                "Required sources: " + ", ".join(source.value for source in self.required_sources)
            )
        return "\n".join(parts)

    def to_checkpoint_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "objective": self.objective,
            "success_criteria": [
                {
                    "kind": item.kind,
                    "description": item.description,
                    "source": item.source.value if item.source is not None else None,
                }
                for item in self.success_criteria
            ],
            "constraints": [
                {"kind": item.kind, "value": _json_value(item.value)}
                for item in self.constraints
            ],
            "required_sources": [item.value for item in self.required_sources],
            "resolved_entities": [
                {
                    "surface": item.surface,
                    "canonical": item.canonical,
                    "confidence": item.confidence,
                }
                for item in self.resolved_entities
            ],
            "unresolved_entities": [
                {"surface": item.surface, "reason": item.reason}
                for item in self.unresolved_entities
            ],
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "parent_goal_id": self.parent_goal_id,
        }

    @classmethod
    def from_checkpoint_dict(cls, payload: object) -> GoalContract:
        if not isinstance(payload, Mapping):
            raise ValueError("checkpoint goal must be an object")
        try:
            criteria = tuple(
                SuccessCriterion(
                    kind=str(item["kind"]),
                    description=str(item.get("description", "")),
                    source=(
                        SourceKind(str(item["source"]))
                        if item.get("source") is not None
                        else None
                    ),
                )
                for item in _mapping_list(payload["success_criteria"], "success_criteria")
            )
            constraints = tuple(
                GoalConstraint(kind=str(item["kind"]), value=item["value"])
                for item in _mapping_list(payload["constraints"], "constraints")
            )
            resolved = tuple(
                ResolvedEntity(
                    surface=str(item["surface"]),
                    canonical=str(item["canonical"]),
                    confidence=float(item["confidence"]),
                )
                for item in _mapping_list(payload["resolved_entities"], "resolved_entities")
            )
            unresolved = tuple(
                UnresolvedEntity(surface=str(item["surface"]), reason=str(item["reason"]))
                for item in _mapping_list(payload["unresolved_entities"], "unresolved_entities")
            )
            required_sources = tuple(
                SourceKind(str(value))
                for value in _list_value(payload["required_sources"], "required_sources")
            )
            return cls(
                goal_id=str(payload["goal_id"]),
                objective=str(payload["objective"]),
                success_criteria=criteria,
                constraints=constraints,
                required_sources=required_sources,
                resolved_entities=resolved,
                unresolved_entities=unresolved,
                status=GoalStatus(str(payload["status"])),
                created_at=str(payload["created_at"]),
                updated_at=str(payload["updated_at"]),
                parent_goal_id=(
                    str(payload["parent_goal_id"])
                    if payload.get("parent_goal_id") is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid checkpoint goal") from exc


@dataclass(frozen=True)
class TurnBudget:
    max_transitions: int = 12
    max_planned_actions: int = 4
    max_tool_calls: int = 4
    max_model_calls: int = 4
    max_planner_calls: int = 2
    max_retrieval_rounds: int = 2
    max_structured_repairs: int = 1
    max_answer_repairs: int = 1
    max_readonly_tool_retries: int = 1
    max_same_action_failures: int = 2
    soft_deadline_ms: int | None = None
    hard_deadline_ms: int | None = 120_000

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value is None and name in {"soft_deadline_ms", "hard_deadline_ms"}:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            self.soft_deadline_ms is not None
            and self.hard_deadline_ms is not None
            and self.soft_deadline_ms > self.hard_deadline_ms
        ):
            raise ValueError("soft deadline must not exceed hard deadline")


@dataclass(frozen=True)
class PlannedAction:
    action_id: str
    capability: Capability
    tool_name: str | None
    arguments: Mapping[str, JSONValue]
    purpose: str
    expected_observation: str
    required: bool
    side_effect: SideEffectClass = SideEffectClass.NONE
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        if not self.action_id.strip() or not self.purpose.strip():
            raise ValueError("planned action identity, capability and purpose are required")
        object.__setattr__(self, "arguments", _freeze_mapping(self.arguments))


@dataclass(frozen=True)
class Observation:
    action_id: str
    status: ToolExecutionStatus
    data: Mapping[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    retryable: bool = False
    committed: bool = False
    receipt: str | None = None

    def __post_init__(self) -> None:
        if not self.action_id.strip():
            raise ValueError("observation action id and status are required")
        object.__setattr__(self, "data", _freeze_mapping(self.data))


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    source: SourceKind
    content: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            raise ValueError("evidence id must not be empty")
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))


@dataclass(frozen=True)
class EvidenceSet:
    status: EvidenceStatus = EvidenceStatus.NOT_REQUESTED
    records: tuple[EvidenceRecord, ...] = ()
    reason_code: str = ""


@dataclass(frozen=True)
class VerificationReport:
    passed: bool
    reason_code: str
    unmet_criteria: tuple[str, ...] = ()
    supported_evidence_ids: tuple[str, ...] = ()
    repairable: bool = False


@dataclass(frozen=True)
class CancellationState:
    requested: bool = False
    reason_code: str = ""


@dataclass(frozen=True)
class NodeTrace:
    node: TurnNode
    started_at: str
    finished_at: str | None = None
    reason_code: str = ""


@dataclass(frozen=True)
class TerminalOutcome:
    status: TerminalStatus
    final_text: str
    goal_status: GoalStatus
    unmet_criteria: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    recoverable: bool = False
    error_code: str | None = None
    route: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "final_text", self.final_text.strip())
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class TurnState:
    run_id: str
    session_id: str
    source: TurnSource
    goal: GoalContract
    node: TurnNode
    plan: tuple[PlannedAction, ...] = ()
    observations: tuple[Observation, ...] = ()
    evidence: EvidenceSet = field(default_factory=EvidenceSet)
    draft_answer: str | None = None
    verification: VerificationReport | None = None
    budget: TurnBudget = field(default_factory=TurnBudget)
    cancellation: CancellationState = field(default_factory=CancellationState)
    terminal: TerminalOutcome | None = None
    trace: tuple[NodeTrace, ...] = ()

    def __post_init__(self) -> None:
        if not self.run_id.strip() or not self.session_id.strip():
            raise ValueError("turn run id and session id must not be empty")


@dataclass(frozen=True)
class StatePatch:
    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _freeze_mapping(self.values))


@dataclass(frozen=True)
class UsageDelta:
    transitions: int = 0
    planned_actions: int = 0
    tool_calls: int = 0
    model_calls: int = 0
    planner_calls: int = 0
    retrieval_rounds: int = 0
    structured_repairs: int = 0
    answer_repairs: int = 0
    retries: int = 0


@dataclass(frozen=True)
class Advance:
    next_node: TurnNode
    patch: StatePatch = field(default_factory=StatePatch)
    events: tuple[WorkflowEvent, ...] = ()
    usage: UsageDelta = field(default_factory=UsageDelta)


@dataclass(frozen=True)
class Retry:
    target_node: TurnNode
    error_code: str
    patch: StatePatch = field(default_factory=StatePatch)
    events: tuple[WorkflowEvent, ...] = ()
    usage: UsageDelta = field(default_factory=UsageDelta)


@dataclass(frozen=True)
class Terminal:
    outcome: TerminalOutcome
    events: tuple[WorkflowEvent, ...] = ()
    usage: UsageDelta = field(default_factory=UsageDelta)


NodeOutcome: TypeAlias = Advance | Retry | Terminal
