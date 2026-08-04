from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from soca.llm import LLMEngine, StructuredLLMEngine

from .checkpoint import (
    GoalCheckpointStore,
    WorkflowRunCheckpoint,
    now_checkpoint_time,
)
from .contracts import (
    GoalConstraint,
    GoalContract,
    GoalStatus,
    SourceKind,
    SuccessCriterion,
    UnresolvedEntity,
)

if TYPE_CHECKING:
    from .runner import WorkflowRun

_SUCCESS_CRITERIA = frozenset(
    {
        "knowledge_queried",
        "memory_queried",
        "tool_observation_available",
    }
)


class GoalDecisionKind(StrEnum):
    NEW = "new_goal"
    CONTINUE = "continue_goal"
    CORRECT = "correct_goal"
    CANCEL = "cancel_goal"
    SMALLTALK = "smalltalk"


@dataclass(frozen=True)
class GoalDecision:
    kind: GoalDecisionKind
    objective: str
    success_criteria: tuple[SuccessCriterion, ...] = ()
    required_sources: tuple[SourceKind, ...] = ()
    constraints: tuple[GoalConstraint, ...] = ()
    unresolved_entities: tuple[UnresolvedEntity, ...] = ()
    confidence: float = 1.0
    clarification_question: str = ""
    model_calls: int = 0


@dataclass(frozen=True)
class GoalResolution:
    goal: GoalContract
    decision: GoalDecision
    continued: bool
    clarification_needed: bool = False


class GoalResolutionError(ValueError):
    pass


class ActiveGoalStore:
    def __init__(
        self,
        *,
        checkpoint_store: GoalCheckpointStore | None = None,
        session_id: str = "default",
    ) -> None:
        self._lock = threading.RLock()
        self._checkpoint_store = checkpoint_store
        self._session_id = session_id
        self._last_run: WorkflowRunCheckpoint | None = None
        self._checkpoint_revision: int | None = None
        self._checkpoint_digest: str | None = None
        if checkpoint_store is None:
            self._goal = None
        else:
            checkpoint = checkpoint_store.load(session_id)
            self._goal = checkpoint.goal
            self._last_run = checkpoint.last_run
            self._checkpoint_revision = checkpoint.revision or None
            self._checkpoint_digest = checkpoint.digest

    @property
    def current(self) -> GoalContract | None:
        with self._lock:
            return self._goal

    @property
    def last_run(self) -> WorkflowRunCheckpoint | None:
        with self._lock:
            return self._last_run

    def set(self, goal: GoalContract) -> GoalContract:
        with self._lock:
            self._goal = goal
            self._persist()
            return goal

    def clear(self) -> None:
        with self._lock:
            self._goal = None
            self._persist()

    def record_run(self, run: WorkflowRun) -> None:
        with self._lock:
            self._last_run = WorkflowRunCheckpoint(
                run_id=run.state.run_id,
                goal_id=run.state.goal.goal_id,
                terminal_status=run.terminal.status.value,
                updated_at=now_checkpoint_time(),
            )
            self._persist()

    def _persist(self) -> None:
        with self._lock:
            if self._checkpoint_store is not None:
                checkpoint = self._checkpoint_store.save(
                    self._session_id,
                    goal=self._goal,
                    last_run=self._last_run,
                    expected_revision=self._checkpoint_revision,
                    expected_digest=self._checkpoint_digest,
                )
                self._checkpoint_revision = checkpoint.revision
                self._checkpoint_digest = checkpoint.digest


class GoalResolver:
    def __init__(self, store: ActiveGoalStore | None = None) -> None:
        self.store = store or ActiveGoalStore()

    def resolve(
        self,
        text: str,
        *,
        source: Literal["text", "voice", "follow_up"] = "text",
        decision: GoalDecision | None = None,
    ) -> GoalResolution:
        statement = text.strip()
        if not statement:
            raise ValueError("goal text must not be empty")
        active = self.store.current
        selected = decision or GoalDecision(
            kind=GoalDecisionKind.NEW,
            objective=statement,
        )
        if not 0.0 <= selected.confidence <= 1.0:
            raise ValueError("goal decision confidence must be between zero and one")

        if selected.kind is GoalDecisionKind.CANCEL:
            if active is None:
                raise GoalResolutionError("cannot cancel without an active goal")
            goal = GoalContract(
                goal_id=active.goal_id,
                objective=active.objective,
                success_criteria=active.success_criteria,
                constraints=active.constraints,
                required_sources=active.required_sources,
                resolved_entities=active.resolved_entities,
                unresolved_entities=active.unresolved_entities,
                status=GoalStatus.ABANDONED,
                created_at=active.created_at,
                updated_at=datetime.now(UTC).isoformat(),
                parent_goal_id=active.parent_goal_id,
            )
            self.store.clear()
            return GoalResolution(goal, selected, continued=True)

        if selected.kind in {
            GoalDecisionKind.CONTINUE,
            GoalDecisionKind.CORRECT,
        }:
            if active is None:
                raise GoalResolutionError("cannot continue without an active goal")
            objective = (
                selected.objective.strip()
                if selected.kind is GoalDecisionKind.CORRECT
                else active.objective
            )
            if not objective:
                raise GoalResolutionError("corrected objective must not be empty")
            goal = GoalContract(
                goal_id=active.goal_id,
                objective=objective,
                success_criteria=selected.success_criteria or active.success_criteria,
                constraints=active.constraints
                + selected.constraints
                + (GoalConstraint("follow_up", statement),),
                required_sources=selected.required_sources or active.required_sources,
                resolved_entities=active.resolved_entities,
                unresolved_entities=selected.unresolved_entities,
                status=GoalStatus.ACTIVE,
                created_at=active.created_at,
                updated_at=datetime.now(UTC).isoformat(),
                parent_goal_id=active.parent_goal_id,
            )
            self.store.set(goal)
            return GoalResolution(
                goal,
                selected,
                continued=True,
                clarification_needed=bool(selected.clarification_question),
            )

        objective = selected.objective.strip() or statement
        constraints = selected.constraints + (GoalConstraint("turn_source", source),)
        goal = GoalContract(
            goal_id=uuid4().hex,
            objective=objective,
            success_criteria=selected.success_criteria,
            constraints=constraints,
            required_sources=selected.required_sources,
            unresolved_entities=selected.unresolved_entities,
        )
        if selected.kind is not GoalDecisionKind.SMALLTALK:
            self.store.set(goal)
        return GoalResolution(
            goal,
            selected,
            continued=False,
            clarification_needed=bool(selected.clarification_question),
        )


class StructuredGoalResolver:
    def __init__(
        self,
        llm: LLMEngine,
        *,
        repair_attempts: int = 1,
        max_tokens: int = 384,
    ) -> None:
        if repair_attempts not in {0, 1}:
            raise ValueError("goal resolver allows at most one repair")
        if max_tokens < 1:
            raise ValueError("goal resolver max tokens must be positive")
        self.llm = llm
        self.repair_attempts = repair_attempts
        self.max_tokens = max_tokens
        self.last_usage: dict[str, float | int] = {}

    def decide(
        self,
        text: str,
        *,
        active_goal: GoalContract | None,
        working_summary: str = "",
        recent_turns: tuple[str, ...] = (),
        asr_alternatives: tuple[str, ...] = (),
    ) -> GoalDecision:
        prompt = self._prompt(
            text,
            active_goal=active_goal,
            working_summary=working_summary,
            recent_turns=recent_turns[-4:],
            asr_alternatives=asr_alternatives,
        )
        raw = self._generate(prompt)
        try:
            decision = self._parse(raw, model_calls=1)
            self._validate_context(decision, active_goal)
            return decision
        except GoalResolutionError as first_error:
            if self.repair_attempts == 0:
                raise
            repaired = self._generate(
                prompt + "\nValidation error: " + str(first_error) + "\nReturn corrected JSON only."
            )
            decision = self._parse(repaired, model_calls=2)
            self._validate_context(decision, active_goal)
            return decision

    def _generate(self, prompt: str) -> str:
        if isinstance(self.llm, StructuredLLMEngine):
            result = self.llm.generate_structured(
                prompt,
                schema_name="soca_goal_resolution",
                schema=_GOAL_SCHEMA,
                max_tokens=self.max_tokens,
                temperature=0.0,
                top_p=1.0,
                inject_persona=False,
            )
        else:
            result = self.llm.generate(
                prompt,
                max_tokens=self.max_tokens,
                temperature=0.0,
                top_p=1.0,
                inject_persona=False,
            )
        self.last_usage = {
            "prompt_tokens": int(result.n_prompt_tokens),
            "completion_tokens": int(result.n_completion_tokens),
            "ttft_ms": float(result.ttft_ms),
            "total_latency_ms": float(result.total_latency_ms),
            "tokens_per_second": float(result.tokens_per_second),
        }
        return result.text

    @staticmethod
    def _prompt(
        text: str,
        *,
        active_goal: GoalContract | None,
        working_summary: str,
        recent_turns: tuple[str, ...],
        asr_alternatives: tuple[str, ...],
    ) -> str:
        payload = {
            "user_text": text,
            "active_goal": (
                {
                    "goal_id": active_goal.goal_id,
                    "objective": active_goal.objective,
                    "required_sources": [source.value for source in active_goal.required_sources],
                }
                if active_goal is not None
                else None
            ),
            "working_summary": working_summary,
            "recent_turns": list(recent_turns),
            "asr_alternatives": list(asr_alternatives),
        }
        return "\n".join(
            [
                "Resolve the user's conversational goal as structured data.",
                "Classify new, continue, correct, cancel, or smalltalk.",
                "When active_goal is null, kind must be new_goal or smalltalk.",
                "A request that requires any source is not smalltalk.",
                "Do not infer a source unless the user or active goal requires it.",
                "If the user explicitly asks for their notes, set required_sources to knowledge and include success criterion knowledge_queried.",
                "ASR alternatives are candidate transcriptions, not facts. If one resolves an entity, use that candidate in the objective; do not invent entities. If candidates conflict or remain unclear, preserve the unresolved entity and ask a clarification question.",
                "Put ambiguous entities in unresolved_entities and provide a clarification question.",
                "Return JSON only.",
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
            ]
        )

    @staticmethod
    def _parse(raw: str, *, model_calls: int) -> GoalDecision:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GoalResolutionError("invalid_json") from exc
        if not isinstance(payload, dict):
            raise GoalResolutionError("root_not_object")
        try:
            kind = GoalDecisionKind(str(payload["kind"]))
            confidence = float(payload["confidence"])
            sources = tuple(
                SourceKind(str(value)) for value in _string_list(payload, "required_sources")
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GoalResolutionError("invalid_decision_fields") from exc
        objective = payload.get("objective")
        clarification = payload.get("clarification_question")
        if (
            not isinstance(objective, str)
            or not isinstance(clarification, str)
            or not 0.0 <= confidence <= 1.0
        ):
            raise GoalResolutionError("invalid_decision_fields")
        criterion_values = _string_list(payload, "success_criteria")
        if any(value not in _SUCCESS_CRITERIA for value in criterion_values):
            raise GoalResolutionError("unsupported_success_criterion")
        criteria = tuple(SuccessCriterion(kind=value) for value in criterion_values)
        constraints = tuple(
            GoalConstraint(str(item["kind"]), item["value"])
            for item in _object_list(payload, "constraints")
            if "kind" in item and "value" in item
        )
        unresolved = tuple(
            UnresolvedEntity(str(item["surface"]), str(item["reason"]))
            for item in _object_list(payload, "unresolved_entities")
            if "surface" in item and "reason" in item
        )
        if SourceKind.KNOWLEDGE in sources and not any(
            item.kind == "knowledge_queried" for item in criteria
        ):
            raise GoalResolutionError("knowledge_source_without_success_criterion")
        return GoalDecision(
            kind=kind,
            objective=objective,
            success_criteria=criteria,
            required_sources=sources,
            constraints=constraints,
            unresolved_entities=unresolved,
            confidence=confidence,
            clarification_question=clarification,
            model_calls=model_calls,
        )

    @staticmethod
    def _validate_context(
        decision: GoalDecision,
        active_goal: GoalContract | None,
    ) -> None:
        if active_goal is None and decision.kind in {
            GoalDecisionKind.CONTINUE,
            GoalDecisionKind.CORRECT,
            GoalDecisionKind.CANCEL,
        }:
            raise GoalResolutionError("decision_requires_active_goal")
        if decision.kind is GoalDecisionKind.SMALLTALK and (
            decision.required_sources or decision.success_criteria
        ):
            raise GoalResolutionError("smalltalk_cannot_require_sources")


def _string_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise GoalResolutionError(f"invalid_{key}")
    return value


def _object_list(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise GoalResolutionError(f"invalid_{key}")
    return value


_GOAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"enum": [item.value for item in GoalDecisionKind]},
        "objective": {"type": "string"},
        "success_criteria": {
            "type": "array",
            "items": {"enum": sorted(_SUCCESS_CRITERIA)},
        },
        "required_sources": {
            "type": "array",
            "items": {"enum": [item.value for item in SourceKind]},
        },
        "constraints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["kind", "value"],
                "additionalProperties": False,
            },
        },
        "unresolved_entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "surface": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["surface", "reason"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "clarification_question": {"type": "string"},
    },
    "required": [
        "kind",
        "objective",
        "success_criteria",
        "required_sources",
        "constraints",
        "unresolved_entities",
        "confidence",
        "clarification_question",
    ],
    "additionalProperties": False,
}
