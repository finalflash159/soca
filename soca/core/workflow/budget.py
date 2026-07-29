from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .contracts import TurnBudget
from .errors import BudgetExceededError

BudgetKind = Literal["transition", "tool", "model", "retry"]


@dataclass(frozen=True)
class BudgetSnapshot:
    transitions: int
    tool_calls: int
    model_calls: int
    retries: int
    elapsed_ms: float


class BudgetLedger:
    """Mutable per-turn accounting with one enforcement point per budget."""

    def __init__(self, budget: TurnBudget, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.budget = budget
        self._clock = clock
        self._started = clock()
        self._counts = {"transition": 0, "tool": 0, "model": 0, "retry": 0}

    def consume(self, kind: BudgetKind, amount: int = 1) -> BudgetSnapshot:
        if amount < 0:
            raise ValueError("budget amount must be non-negative")
        limit = {
            "transition": self.budget.max_transitions,
            "tool": self.budget.max_tool_calls,
            "model": self.budget.max_model_calls,
            "retry": self.budget.max_retries,
        }[kind]
        next_value = self._counts[kind] + amount
        if next_value > limit:
            raise BudgetExceededError(kind)
        self._counts[kind] = next_value
        snapshot = self.snapshot()
        if (
            self.budget.max_elapsed_ms is not None
            and snapshot.elapsed_ms > self.budget.max_elapsed_ms
        ):
            raise BudgetExceededError("elapsed_ms")
        return snapshot

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            transitions=self._counts["transition"],
            tool_calls=self._counts["tool"],
            model_calls=self._counts["model"],
            retries=self._counts["retry"],
            elapsed_ms=(self._clock() - self._started) * 1000,
        )
