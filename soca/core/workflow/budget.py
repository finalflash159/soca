from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .contracts import TurnBudget
from .errors import BudgetExceededError

BudgetKind = Literal[
    "transition",
    "planned_action",
    "tool",
    "model",
    "planner",
    "retrieval_round",
    "structured_repair",
    "answer_repair",
    "retry",
]


@dataclass(frozen=True)
class BudgetSnapshot:
    transitions: int
    planned_actions: int
    tool_calls: int
    model_calls: int
    planner_calls: int
    retrieval_rounds: int
    structured_repairs: int
    answer_repairs: int
    retries: int
    elapsed_ms: float


class BudgetLedger:
    """Mutable per-turn accounting with one enforcement point per budget."""

    def __init__(self, budget: TurnBudget, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.budget = budget
        self._clock = clock
        self._started = clock()
        self._counts = {
            "transition": 0,
            "planned_action": 0,
            "tool": 0,
            "model": 0,
            "planner": 0,
            "retrieval_round": 0,
            "structured_repair": 0,
            "answer_repair": 0,
            "retry": 0,
        }

    def consume(self, kind: BudgetKind, amount: int = 1) -> BudgetSnapshot:
        if amount < 0:
            raise ValueError("budget amount must be non-negative")
        limit = {
            "transition": self.budget.max_transitions,
            "planned_action": self.budget.max_planned_actions,
            "tool": self.budget.max_tool_calls,
            "model": self.budget.max_model_calls,
            "planner": self.budget.max_planner_calls,
            "retrieval_round": self.budget.max_retrieval_rounds,
            "structured_repair": self.budget.max_structured_repairs,
            "answer_repair": self.budget.max_answer_repairs,
            "retry": self.budget.max_readonly_tool_retries,
        }[kind]
        next_value = self._counts[kind] + amount
        if next_value > limit:
            raise BudgetExceededError(kind)
        self._counts[kind] = next_value
        snapshot = self.snapshot()
        if (
            self.budget.hard_deadline_ms is not None
            and snapshot.elapsed_ms > self.budget.hard_deadline_ms
        ):
            raise BudgetExceededError("elapsed_ms")
        return snapshot

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            transitions=self._counts["transition"],
            planned_actions=self._counts["planned_action"],
            tool_calls=self._counts["tool"],
            model_calls=self._counts["model"],
            planner_calls=self._counts["planner"],
            retrieval_rounds=self._counts["retrieval_round"],
            structured_repairs=self._counts["structured_repair"],
            answer_repairs=self._counts["answer_repair"],
            retries=self._counts["retry"],
            elapsed_ms=(self._clock() - self._started) * 1000,
        )
