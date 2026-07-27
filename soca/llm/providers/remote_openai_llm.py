"""OpenAI-compatible remote LLM engine.

One class serves every provider in :mod:`provider_registry` (OpenAI, Groq,
OpenRouter, Gemini) because they all speak the OpenAI `/chat/completions`
dialect. It implements the same :class:`~soca.llm.base.LLMEngine` Protocol as
the local llama.cpp runner, so nothing downstream (runtime, guardrail, voice
loop) needs to change to use it.

The OpenAI client is injectable so tests drive it without any network access.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from typing import Any

from soca.prompts import SOCA_LLM_SYSTEM_PROMPT, split_embedded_system_prompt

from ..base import LLMResult
from ..message_format import ChatMessage
from .provider_registry import LLMProvider

DEFAULT_TIMEOUT_S = 60.0
DEFAULT_MAX_RETRIES = 2
# Rough fallback when no tokenizer is available: ~4 chars per token.
_CHARS_PER_TOKEN = 4


class RemoteLLMError(RuntimeError):
    """A remote provider call failed, with a user-friendly Vietnamese message.

    ``category`` is one of ``auth`` | ``rate_limit`` | ``network`` | ``unknown``
    so UI code can react without parsing the message text.
    """

    def __init__(self, message: str, *, category: str = "unknown") -> None:
        super().__init__(message)
        self.category = category


def build_remote_messages(user_msg: str, inject_persona: bool) -> list[ChatMessage]:
    """SoCa persona + user turn as OpenAI chat messages.

    Mirrors :func:`soca.llm.message_format.build_chat_messages` but without the
    GGUF-specific ``LLMModelConfig`` coupling: remote chat models need only a
    system persona and the user message.
    """
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


def _map_error(exc: Exception, provider: LLMProvider) -> RemoteLLMError:
    """Translate an OpenAI/transport exception into a friendly RemoteLLMError.

    Matches on ``status_code`` and class name so it works both for the real
    ``openai`` exception types and for lightweight fakes in tests.
    """
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    label = provider.label

    if status == 401 or status == 403 or "Authentication" in name or "Permission" in name:
        return RemoteLLMError(
            f"API key {label} sai hoặc hết hạn.", category="auth"
        )
    if status == 429 or "RateLimit" in name:
        return RemoteLLMError(
            f"{label} đã hết quota hoặc bị giới hạn tốc độ (rate limit).",
            category="rate_limit",
        )
    if "Connection" in name or "Timeout" in name:
        return RemoteLLMError(
            f"Không kết nối được tới {label}. Kiểm tra mạng và thử lại.",
            category="network",
        )
    return RemoteLLMError(f"Lỗi khi gọi {label}: {exc}", category="unknown")


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
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if not api_key.strip():
            raise ValueError("api_key must not be empty")

        self.provider = provider
        self.model = model
        self._client = client if client is not None else _build_client(
            provider, api_key, timeout=timeout, max_retries=max_retries
        )

    # -- validation ---------------------------------------------------------

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
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def _create_non_streaming_result(
        self,
        messages: list[ChatMessage],
        request: dict[str, Any],
    ) -> LLMResult:
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:  # noqa: BLE001 - provider boundary translation
            raise _map_error(exc, self.provider) from exc
        ended = time.perf_counter()

        text = (response.choices[0].message.content or "").strip()
        usage = getattr(response, "usage", None)
        n_prompt = getattr(usage, "prompt_tokens", 0) or 0
        n_completion = getattr(usage, "completion_tokens", 0) or 0
        total_latency_ms = (ended - started) * 1000
        elapsed = ended - started
        tokens_per_second = n_completion / elapsed if elapsed > 0 else 0.0
        return LLMResult(
            text=text,
            prompt=self._prompt_for_metrics(messages),
            n_prompt_tokens=n_prompt,
            n_completion_tokens=n_completion,
            ttft_ms=total_latency_ms,
            total_latency_ms=total_latency_ms,
            tokens_per_second=tokens_per_second,
        )

    # -- generation ---------------------------------------------------------

    def generate(
        self,
        user_msg: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        inject_persona: bool = True,
    ) -> LLMResult:
        self._validate_generation_args(user_msg, max_tokens, temperature, top_p)
        messages = build_remote_messages(user_msg, inject_persona)
        return self._create_non_streaming_result(
            messages,
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False,
            },
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
        if not isinstance(schema_name, str) or not schema_name.strip():
            raise ValueError("structured output schema name is invalid")
        if not isinstance(schema, Mapping):
            raise ValueError("structured output schema is invalid")

        messages = build_remote_messages(user_msg, inject_persona)
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
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
        }
        if self.provider.key == "openrouter":
            request["extra_body"] = {
                "provider": {
                    "require_parameters": True,
                    "data_collection": "deny" if zero_data_retention else "allow",
                }
            }
        return self._create_non_streaming_result(messages, request)

    def generate_stream(
        self,
        user_msg: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        inject_persona: bool = True,
    ) -> Iterator[str]:
        self._validate_generation_args(user_msg, max_tokens, temperature, top_p)
        messages = build_remote_messages(user_msg, inject_persona)

        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                text = _chunk_delta_text(chunk)
                if text:
                    yield text
        except Exception as exc:  # noqa: BLE001 - boundary: translate to friendly error
            raise _map_error(exc, self.provider) from exc

    def count_tokens(self, text: str) -> int:
        """Pre-send token estimate (usage from responses is authoritative).

        Uses a ~4-chars-per-token heuristic; exact counts come back in the
        response ``usage`` for both generate and streaming calls.
        """
        if not text:
            return 0
        return max(1, round(len(text) / _CHARS_PER_TOKEN))


def _chunk_delta_text(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    return (getattr(delta, "content", None) or "") if delta is not None else ""


def _build_client(
    provider: LLMProvider,
    api_key: str,
    *,
    timeout: float,
    max_retries: int,
) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RemoteLLMError(
            "Thiếu gói 'openai'. Cài đặt: pip install 'soca[llm-remote]'.",
            category="unknown",
        ) from exc

    return OpenAI(
        base_url=provider.base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )


__all__ = ["RemoteLLMError", "RemoteOpenAILLM", "build_remote_messages"]
