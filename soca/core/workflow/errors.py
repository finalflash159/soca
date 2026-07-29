from __future__ import annotations

from enum import StrEnum


class WorkflowErrorCode(StrEnum):
    INVALID_GOAL = "invalid_goal"
    BUDGET_EXHAUSTED = "budget_exhausted"
    DUPLICATE_TERMINAL = "duplicate_terminal"
    CANCELLED = "cancelled"
    MODEL_ERROR = "model_error"
    TOOL_ERROR = "tool_error"
    PROTOCOL_ERROR = "protocol_error"
    WORKER_ERROR = "worker_error"


class WorkflowError(Exception):
    def __init__(self, code: WorkflowErrorCode, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class BudgetExceededError(WorkflowError):
    def __init__(self, budget_name: str):
        super().__init__(
            WorkflowErrorCode.BUDGET_EXHAUSTED,
            f"workflow budget exhausted: {budget_name}",
        )
        self.budget_name = budget_name


class DuplicateTerminalError(WorkflowError):
    def __init__(self) -> None:
        super().__init__(
            WorkflowErrorCode.DUPLICATE_TERMINAL,
            "workflow terminal outcome was already emitted",
        )


class WorkflowCancelledError(WorkflowError):
    def __init__(self) -> None:
        super().__init__(WorkflowErrorCode.CANCELLED, "workflow was cancelled")
