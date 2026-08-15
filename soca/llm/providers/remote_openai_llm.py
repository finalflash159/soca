from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from soca.prompts import SOCA_LLM_SYSTEM_PROMPT, split_embedded_system_prompt

from ..base import LLMResult
from ..message_format import ChatMessage
from .provider_registry import LLMProvider
from .request_adapter import ProviderRequestAdapter, ReasoningParameter
from .response_adapter import (
    chunk_fields,
    close_stream,
    empty_response_error,
    empty_stream_error,
    first_choice,
    map_provider_error,
    message_text,
    output_limit_error,
)
from .runtime_contracts import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_S,
    ProviderAttempt,
    ProviderCallTrace,
    RemoteFailureKind,
    RemoteLLMError,
    RetryPolicy,
    replace_remote_error,
    trace_with_response_failure,
)

_CHARS_PER_TOKEN = 4


@dataclass
class _CallState:
    cancelled: threading.Event = field(default_factory=threading.Event)
    stream: Any | None = None


def build_remote_messages(user_msg: str, inject_persona: bool) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    if inject_persona:
        messages.append({"role": "system", "content": SOCA_LLM_SYSTEM_PROMPT})
        user_content = user_msg.strip()
    else:
        system_prompt, user_content = split_embedded_system_prompt(user_msg)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content.strip()})
    return messages


class RemoteOpenAILLM:
    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        api_key: str,
        *,
        client: Any | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        reasoning_enabled: bool | None = None,
        reasoning_parameter: ReasoningParameter | None = None,
        max_output_tokens: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if max_output_tokens is not None and max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")

        self.provider = provider
        self.model = model
        self.reasoning_enabled: bool | None = reasoning_enabled
        self.reasoning_parameter: ReasoningParameter | None = reasoning_parameter
        self.max_output_tokens = max_output_tokens
        self._adapter = ProviderRequestAdapter(provider)
        self._policy = RetryPolicy(max_attempts=max_retries + 1, deadline_s=timeout)
        self._sleep = sleep
        self._clock = clock
        self._state_lock = threading.Lock()
        self._active_call: _CallState | None = None
        self.last_call_trace: ProviderCallTrace | None = None
        self._client = (
            client if client is not None else _build_client(provider, api_key, timeout=timeout)
        )

    @staticmethod
    def _validate_generation_args(
        user_msg: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        if not user_msg.strip():
            raise ValueError("user_msg must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in the range (0, 1]")

    @staticmethod
    def _prompt_for_metrics(messages: list[ChatMessage]) -> str:
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)

    def _effective_max_tokens(self, requested: int) -> int:
        if self.max_output_tokens is None:
            return requested
        return min(requested, self.max_output_tokens)

    def _request_options(self, max_tokens: int) -> dict[str, Any]:
        return self._adapter.generation_options(
            max_tokens=max_tokens,
            reasoning_enabled=self.reasoning_enabled,
            reasoning_parameter=self.reasoning_parameter,
        )

    def _begin_call(self) -> _CallState:
        with self._state_lock:
            if self._active_call is not None:
                raise RuntimeError("concurrent calls on one LLM engine are not supported")
            state = _CallState()
            self._active_call = state
            return state

    def _end_call(self, state: _CallState) -> None:
        with self._state_lock:
            if self._active_call is state:
                self._active_call = None

    def cancel(self) -> None:
        with self._state_lock:
            state = self._active_call
            if state is None:
                return
            state.cancelled.set()
            stream = state.stream
        close_stream(stream)

    def _raise_if_cancelled(self, state: _CallState) -> None:
        if state.cancelled.is_set():
            raise RemoteLLMError(
                "Yêu cầu tới model đã bị hủy.",
                category=RemoteFailureKind.CANCELLED,
                provider=self.provider.key,
                model=self.model,
            )

    def _call_with_retry(
        self,
        operation: str,
        request: Mapping[str, Any],
        *,
        requested_max_tokens: int,
        effective_max_tokens: int,
        state: _CallState,
    ) -> tuple[Any, ProviderCallTrace]:
        started = self._clock()
        attempts: list[ProviderAttempt] = []
        for attempt in range(1, self._policy.max_attempts + 1):
            self._raise_if_cancelled(state)
            try:
                response = self._client.chat.completions.create(**dict(request))
            except Exception as exc:
                mapped = map_provider_error(exc, self.provider, self.model)
                if mapped is None:
                    raise
                elapsed = self._clock() - started
                can_retry = mapped.retryable and attempt < self._policy.max_attempts
                delay = self._policy.delay(attempt, mapped.retry_after_s) if can_retry else 0.0
                if can_retry and elapsed + delay < self._policy.deadline_s:
                    attempts.append(
                        ProviderAttempt(
                            attempt,
                            "failure",
                            mapped.category,
                            mapped.status_code,
                            delay,
                        )
                    )
                    self._sleep(delay)
                    continue
                attempts.append(
                    ProviderAttempt(
                        attempt,
                        "cancelled" if mapped.category == "cancelled" else "failure",
                        mapped.category,
                        mapped.status_code,
                    )
                )
                trace = self._make_trace(
                    operation,
                    requested_max_tokens,
                    effective_max_tokens,
                    attempts,
                    started,
                )
                self.last_call_trace = trace
                raise mapped.with_attempts(attempt) from exc
            attempts.append(ProviderAttempt(attempt, "success"))
            trace = self._make_trace(
                operation,
                requested_max_tokens,
                effective_max_tokens,
                attempts,
                started,
            )
            self.last_call_trace = trace
            return response, trace
        raise AssertionError("retry loop exited without a result")

    def _make_trace(
        self,
        operation: str,
        requested_max_tokens: int,
        effective_max_tokens: int,
        attempts: list[ProviderAttempt],
        started: float,
    ) -> ProviderCallTrace:
        return ProviderCallTrace(
            provider=self.provider.key,
            model=self.model,
            operation=operation,
            requested_max_tokens=requested_max_tokens,
            effective_max_tokens=effective_max_tokens,
            attempts=tuple(attempts),
            elapsed_ms=(self._clock() - started) * 1000,
        )

    def _create_non_streaming_result(
        self,
        messages: list[ChatMessage],
        request: dict[str, Any],
        *,
        requested_max_tokens: int,
        effective_max_tokens: int,
        operation: str,
    ) -> LLMResult:
        state = self._begin_call()
        started = time.perf_counter()
        try:
            response, trace = self._call_with_retry(
                operation,
                request,
                requested_max_tokens=requested_max_tokens,
                effective_max_tokens=effective_max_tokens,
                state=state,
            )
            self._raise_if_cancelled(state)
        finally:
            self._end_call(state)
        ended = time.perf_counter()

        try:
            choice, message = first_choice(response, self.provider, self.model)
            text = message_text(message)
            finish_reason = str(getattr(choice, "finish_reason", None) or "")
            if not text:
                raise empty_response_error(choice, message, self.provider, self.model)
            if finish_reason == "length":
                raise output_limit_error(self.provider, self.model)
        except RemoteLLMError as error:
            self.last_call_trace = trace_with_response_failure(trace, error)
            raise
        usage = getattr(response, "usage", None)
        n_prompt = getattr(usage, "prompt_tokens", 0) or 0
        n_completion = getattr(usage, "completion_tokens", 0) or 0
        elapsed = ended - started
        return LLMResult(
            text=text,
            prompt=self._prompt_for_metrics(messages),
            n_prompt_tokens=n_prompt,
            n_completion_tokens=n_completion,
            ttft_ms=elapsed * 1000,
            total_latency_ms=elapsed * 1000,
            tokens_per_second=n_completion / elapsed if elapsed > 0 else 0.0,
            provider_trace=trace.as_dict(),
        )

    def generate(
        self,
        user_msg: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        inject_persona: bool = True,
    ) -> LLMResult:
        self._validate_generation_args(user_msg, max_tokens, temperature, top_p)
        effective = self._effective_max_tokens(max_tokens)
        messages = build_remote_messages(user_msg, inject_persona)
        request = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
            **self._request_options(effective),
        }
        return self._create_non_streaming_result(
            messages,
            request,
            requested_max_tokens=max_tokens,
            effective_max_tokens=effective,
            operation="generate",
        )

    def generate_structured(
        self,
        user_msg: str,
        *,
        schema_name: str,
        schema: Mapping[str, Any],
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        inject_persona: bool = False,
        zero_data_retention: bool = True,
    ) -> LLMResult:
        self._validate_generation_args(user_msg, max_tokens, temperature, top_p)
        if not schema_name.strip():
            raise ValueError("structured output schema name is invalid")
        if not isinstance(schema, Mapping):
            raise ValueError("structured output schema is invalid")
        effective = self._effective_max_tokens(max_tokens)
        messages = build_remote_messages(user_msg, inject_persona)
        options = self._adapter.merge_options(
            self._request_options(effective),
            self._adapter.structured_options(zero_data_retention=zero_data_retention),
        )
        request = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": dict(schema),
                },
            },
            **options,
        }
        return self._create_non_streaming_result(
            messages,
            request,
            requested_max_tokens=max_tokens,
            effective_max_tokens=effective,
            operation="generate_structured",
        )

    def generate_stream(
        self,
        user_msg: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        inject_persona: bool = True,
    ) -> Iterator[str]:
        self._validate_generation_args(user_msg, max_tokens, temperature, top_p)
        effective = self._effective_max_tokens(max_tokens)
        messages = build_remote_messages(user_msg, inject_persona)
        request = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
            **self._request_options(effective),
        }
        state = self._begin_call()
        started = self._clock()
        attempts: list[ProviderAttempt] = []
        try:
            for attempt in range(1, self._policy.max_attempts + 1):
                emitted = False
                finish_reason = ""
                reasoning_seen = False
                refusal = ""
                stream = None
                self._raise_if_cancelled(state)
                try:
                    stream = self._client.chat.completions.create(**request)
                    state.stream = stream
                    for chunk in stream:
                        self._raise_if_cancelled(state)
                        fields = chunk_fields(chunk)
                        chunk_text, chunk_reasoning, chunk_refusal, chunk_finish = fields
                        reasoning_seen = reasoning_seen or chunk_reasoning
                        refusal = refusal or chunk_refusal
                        finish_reason = chunk_finish or finish_reason
                        if chunk_text:
                            emitted = True
                            yield chunk_text
                    self._raise_if_cancelled(state)
                except GeneratorExit:
                    state.cancelled.set()
                    raise
                except RemoteLLMError as error:
                    mapped = error
                except Exception as exc:
                    mapped = map_provider_error(exc, self.provider, self.model)
                    if mapped is None:
                        raise
                else:
                    if emitted:
                        if finish_reason == "length":
                            mapped = output_limit_error(self.provider, self.model)
                        else:
                            attempts.append(ProviderAttempt(attempt, "success"))
                            self.last_call_trace = self._make_trace(
                                "generate_stream", max_tokens, effective, attempts, started
                            )
                            return
                    else:
                        mapped = empty_stream_error(
                            refusal=refusal,
                            reasoning_seen=reasoning_seen,
                            finish_reason=finish_reason,
                            provider=self.provider,
                            model=self.model,
                        )
                finally:
                    close_stream(stream)
                    state.stream = None

                if emitted:
                    # Keep the cause. Replacing it with the generic wording hid
                    # the only actionable fact a caller has — for an output
                    # limit, that the budget must be raised — behind a message
                    # that reads like a network fault.
                    mapped = replace_remote_error(
                        mapped,
                        message=(
                            f"Luồng {self.provider.label} bị ngắt sau một phần nội dung: "
                            f"{mapped.args[0] if mapped.args else mapped.category}"
                        ),
                        retryable=False,
                    )
                elapsed = self._clock() - started
                can_retry = (
                    mapped.retryable
                    and not emitted
                    and attempt < self._policy.max_attempts
                )
                delay = self._policy.delay(attempt, mapped.retry_after_s) if can_retry else 0.0
                if can_retry and elapsed + delay < self._policy.deadline_s:
                    attempts.append(
                        ProviderAttempt(
                            attempt,
                            "failure",
                            mapped.category,
                            mapped.status_code,
                            delay,
                        )
                    )
                    self.last_call_trace = self._make_trace(
                        "generate_stream", max_tokens, effective, attempts, started
                    )
                    self._sleep(delay)
                    continue
                attempts.append(
                    ProviderAttempt(
                        attempt,
                        "cancelled" if mapped.category == "cancelled" else "failure",
                        mapped.category,
                        mapped.status_code,
                    )
                )
                self.last_call_trace = self._make_trace(
                    "generate_stream", max_tokens, effective, attempts, started
                )
                raise mapped.with_attempts(attempt)
        finally:
            self._end_call(state)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, round(len(text) / _CHARS_PER_TOKEN))


def _build_client(provider: LLMProvider, api_key: str, *, timeout: float) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise RemoteLLMError(
            "Thiếu gói 'openai'. Cài đặt: pip install 'soca[llm-remote]'.",
            category=RemoteFailureKind.REQUEST,
            provider=provider.key,
        ) from exc
    return OpenAI(
        base_url=provider.base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=0,
    )


__all__ = [
    "ProviderAttempt",
    "ProviderCallTrace",
    "RemoteFailureKind",
    "RemoteLLMError",
    "RemoteOpenAILLM",
    "RetryPolicy",
    "build_remote_messages",
]
