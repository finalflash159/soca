from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import uuid4

from .contracts import GoalContract


@dataclass(frozen=True)
class GoalResolution:
    goal: GoalContract
    continued: bool
    clarification_needed: bool = False


class ActiveGoalStore:
    def __init__(self) -> None:
        self._goal: GoalContract | None = None

    @property
    def current(self) -> GoalContract | None:
        return self._goal

    def set(self, goal: GoalContract) -> GoalContract:
        self._goal = goal
        return goal

    def clear(self) -> None:
        self._goal = None


class GoalResolver:
    """Resolve a turn while keeping continuation decisions explicit."""

    def __init__(self, store: ActiveGoalStore | None = None) -> None:
        self.store = store or ActiveGoalStore()

    def resolve(
        self,
        text: str,
        *,
        source: Literal["text", "voice", "follow_up"] = "text",
        continues_active_goal: bool = False,
    ) -> GoalResolution:
        statement = text.strip()
        if not statement:
            raise ValueError("goal text must not be empty")
        active = self.store.current
        if continues_active_goal and active is None:
            raise ValueError("cannot continue without an active goal")
        if continues_active_goal and active is not None:
            goal = GoalContract(
                statement=active.statement,
                goal_id=active.goal_id,
                source="follow_up",
                success_criteria=active.success_criteria,
                metadata={"follow_up": statement, "source": source},
            )
            self.store.set(goal)
            return GoalResolution(goal=goal, continued=True)

        goal = GoalContract(statement=statement, goal_id=uuid4().hex, source=source)
        self.store.set(goal)
        return GoalResolution(goal=goal, continued=False)
