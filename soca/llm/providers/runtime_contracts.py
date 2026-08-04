from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

DEFAULT_TIMEOUT_S = 60.0
DEFAULT_MAX_RETRIES = 2


class RemoteFailureKind(StrEnum):
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    NETWORK = "network"
    SERVER = "server"
    REQUEST = "request"
    MODEL = "model"
    REFUSAL = "refusal"
    OUTPUT_LIMIT = "output_limit"
    REASONING_ONLY = "reasoning_only"
    EMPTY_RESPONSE = "empty_response"
    PROTOCOL = "protocol"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class RemoteLLMError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        category: str = RemoteFailureKind.UNKNOWN,
        provider: str = "",
        model: str = "",
        status_code: int | None = None,
        provider_code: str = "",
        retryable: bool = False,
        attempts: int = 1,
        finish_reason: str = "",
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.category = str(category)
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.provider_code = provider_code
        self.retryable = retryable
        self.attempts = attempts
        self.finish_reason = finish_reason
        self.retry_after_s = retry_after_s

    def with_attempts(self, attempts: int) -> RemoteLLMError:
        return RemoteLLMError(
            str(self),
            category=self.category,
            provider=self.provider,
            model=self.model,
            status_code=self.status_code,
            provider_code=self.provider_code,
            retryable=self.retryable,
            attempts=attempts,
            finish_reason=self.finish_reason,
            retry_after_s=self.retry_after_s,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "message": str(self),
            "category": self.category,
            "provider": self.provider,
            "model": self.model,
            "status_code": self.status_code,
            "provider_code": self.provider_code,
            "retryable": self.retryable,
            "attempts": self.attempts,
            "finish_reason": self.finish_reason,
            "retry_after_s": self.retry_after_s,
        }


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = DEFAULT_MAX_RETRIES + 1
    base_delay_s: float = 0.25
    max_delay_s: float = 4.0
    deadline_s: float = DEFAULT_TIMEOUT_S

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay_s < 0 or self.max_delay_s < 0 or self.deadline_s <= 0:
            raise ValueError("retry timing values are invalid")

    def delay(self, failed_attempt: int, retry_after_s: float | None) -> float:
        if retry_after_s is not None:
            return min(max(0.0, retry_after_s), self.max_delay_s)
        exponential = self.base_delay_s * (2 ** max(0, failed_attempt - 1))
        return min(exponential, self.max_delay_s)


@dataclass(frozen=True)
class ProviderAttempt:
    attempt: int
    outcome: Literal["success", "failure", "cancelled"]
    failure_kind: str = ""
    status_code: int | None = None
    retry_delay_s: float = 0.0


@dataclass(frozen=True)
class ProviderCallTrace:
    provider: str
    model: str
    operation: str
    requested_max_tokens: int
    effective_max_tokens: int
    attempts: tuple[ProviderAttempt, ...]
    elapsed_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "operation": self.operation,
            "requested_max_tokens": self.requested_max_tokens,
            "effective_max_tokens": self.effective_max_tokens,
            "attempt_count": len(self.attempts),
            "retry_count": max(0, len(self.attempts) - 1),
            "attempts": [
                {
                    "attempt": item.attempt,
                    "outcome": item.outcome,
                    "failure_kind": item.failure_kind,
                    "status_code": item.status_code,
                    "retry_delay_s": item.retry_delay_s,
                }
                for item in self.attempts
            ],
            "elapsed_ms": self.elapsed_ms,
        }


def replace_remote_error(
    error: RemoteLLMError,
    *,
    message: str,
    retryable: bool,
) -> RemoteLLMError:
    return RemoteLLMError(
        message,
        category=error.category,
        provider=error.provider,
        model=error.model,
        status_code=error.status_code,
        provider_code=error.provider_code,
        retryable=retryable,
        attempts=error.attempts,
        finish_reason=error.finish_reason,
        retry_after_s=error.retry_after_s,
    )


def trace_with_response_failure(
    trace: ProviderCallTrace,
    error: RemoteLLMError,
) -> ProviderCallTrace:
    attempts = trace.attempts
    if attempts:
        attempts = (
            *attempts[:-1],
            ProviderAttempt(
                attempts[-1].attempt,
                "failure",
                error.category,
                error.status_code,
            ),
        )
    return ProviderCallTrace(
        provider=trace.provider,
        model=trace.model,
        operation=trace.operation,
        requested_max_tokens=trace.requested_max_tokens,
        effective_max_tokens=trace.effective_max_tokens,
        attempts=attempts,
        elapsed_ms=trace.elapsed_ms,
    )


__all__ = [
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_TIMEOUT_S",
    "ProviderAttempt",
    "ProviderCallTrace",
    "RemoteFailureKind",
    "RemoteLLMError",
    "RetryPolicy",
    "replace_remote_error",
    "trace_with_response_failure",
]
