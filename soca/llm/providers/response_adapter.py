from __future__ import annotations

import time
from collections.abc import Mapping
from email.utils import parsedate_to_datetime
from typing import Any

from .provider_registry import LLMProvider
from .runtime_contracts import RemoteFailureKind, RemoteLLMError


def empty_stream_error(
    *,
    refusal: str,
    reasoning_seen: bool,
    finish_reason: str,
    provider: LLMProvider,
    model: str,
) -> RemoteLLMError:
    if refusal:
        return RemoteLLMError(
            f"{provider.label} từ chối yêu cầu: {refusal}",
            category=RemoteFailureKind.REFUSAL,
            provider=provider.key,
            model=model,
            finish_reason=finish_reason,
        )
    if reasoning_seen:
        return RemoteLLMError(
            f"{provider.label} chỉ trả reasoning, không có câu trả lời cuối.",
            category=RemoteFailureKind.REASONING_ONLY,
            provider=provider.key,
            model=model,
            finish_reason=finish_reason,
        )
    return empty_response_error(
        type("Choice", (), {"finish_reason": finish_reason})(),
        None,
        provider,
        model,
    )


def first_choice(response: Any, provider: LLMProvider, model: str) -> tuple[Any, Any]:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise RemoteLLMError(
            f"{provider.label} trả về response không có lựa chọn.",
            category=RemoteFailureKind.PROTOCOL,
            provider=provider.key,
            model=model,
        )
    choice = choices[0]
    return choice, getattr(choice, "message", None)


def message_text(message: Any) -> str:
    raw = getattr(message, "content", None)
    return raw.strip() if isinstance(raw, str) else ""


def reasoning_present(message: Any) -> bool:
    for field_name in ("reasoning", "reasoning_content"):
        value = getattr(message, field_name, None)
        if isinstance(value, str) and value.strip():
            return True
        if value not in (None, "", [], {}):
            return True
    return False


def empty_response_error(
    choice: Any,
    message: Any,
    provider: LLMProvider,
    model: str,
) -> RemoteLLMError:
    finish_reason = str(getattr(choice, "finish_reason", None) or "")
    refusal = getattr(message, "refusal", None) if message is not None else None
    if isinstance(refusal, str) and refusal.strip():
        return RemoteLLMError(
            f"{provider.label} từ chối yêu cầu: {refusal.strip()}",
            category=RemoteFailureKind.REFUSAL,
            provider=provider.key,
            model=model,
            finish_reason=finish_reason,
        )
    if message is not None and reasoning_present(message):
        return RemoteLLMError(
            f"{provider.label} chỉ trả reasoning, không có câu trả lời cuối.",
            category=RemoteFailureKind.REASONING_ONLY,
            provider=provider.key,
            model=model,
            finish_reason=finish_reason,
        )
    if finish_reason == "length":
        return output_limit_error(provider, model)
    return RemoteLLMError(
        f"{provider.label} trả về nội dung rỗng (finish_reason={finish_reason or 'missing'}).",
        category=RemoteFailureKind.EMPTY_RESPONSE,
        provider=provider.key,
        model=model,
        finish_reason=finish_reason,
    )


def output_limit_error(provider: LLMProvider, model: str) -> RemoteLLMError:
    return RemoteLLMError(
        f"{provider.label} hết ngân sách output trước khi tạo câu trả lời cuối.",
        category=RemoteFailureKind.OUTPUT_LIMIT,
        provider=provider.key,
        model=model,
        finish_reason="length",
    )


def chunk_fields(chunk: Any) -> tuple[str, bool, str, str]:
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return "", False, "", ""
    choice = choices[0]
    delta = getattr(choice, "delta", None)
    if delta is None:
        return "", False, "", str(getattr(choice, "finish_reason", None) or "")
    content = getattr(delta, "content", None)
    refusal = getattr(delta, "refusal", None)
    return (
        content if isinstance(content, str) else "",
        reasoning_present(delta),
        refusal.strip() if isinstance(refusal, str) else "",
        str(getattr(choice, "finish_reason", None) or ""),
    )


def map_provider_error(
    exc: Exception,
    provider: LLMProvider,
    model: str,
) -> RemoteLLMError | None:
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    module = type(exc).__module__
    recognized_transport = (
        module.startswith(("openai", "httpx", "httpcore"))
        or "Connection" in name
        or "Timeout" in name
        or isinstance(exc, (ConnectionError, TimeoutError))
    )
    if not isinstance(status, int) and not recognized_transport:
        return None

    common = {
        "provider": provider.key,
        "model": model,
        "status_code": status if isinstance(status, int) else None,
        "provider_code": _provider_error_code(exc),
        "retry_after_s": _retry_after_seconds(exc),
    }
    if status in (401, 403) or "Authentication" in name or "Permission" in name:
        return RemoteLLMError(
            f"API key {provider.label} sai, hết hạn hoặc không có quyền.",
            category=RemoteFailureKind.AUTH,
            **common,
        )
    if status == 404:
        return RemoteLLMError(
            f"{provider.label} không tìm thấy model hoặc endpoint đã chọn.",
            category=RemoteFailureKind.MODEL,
            **common,
        )
    if status == 429 or "RateLimit" in name:
        return RemoteLLMError(
            f"{provider.label} đang giới hạn tốc độ hoặc hết quota.",
            category=RemoteFailureKind.RATE_LIMIT,
            retryable=True,
            **common,
        )
    if status in (408, 409) or (isinstance(status, int) and status >= 500):
        return RemoteLLMError(
            f"{provider.label} tạm thời không sẵn sàng (HTTP {status}).",
            category=RemoteFailureKind.SERVER,
            retryable=True,
            **common,
        )
    if status in (400, 422):
        return RemoteLLMError(
            f"{provider.label} từ chối cấu hình request: {exc}",
            category=RemoteFailureKind.REQUEST,
            **common,
        )
    if recognized_transport:
        return RemoteLLMError(
            f"Không kết nối được tới {provider.label}: {exc}",
            category=RemoteFailureKind.NETWORK,
            retryable=True,
            **common,
        )
    return RemoteLLMError(
        f"Lỗi provider {provider.label}: {exc}",
        category=RemoteFailureKind.UNKNOWN,
        **common,
    )


def _provider_error_code(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, Mapping):
        payload = body.get("error", body)
        if isinstance(payload, Mapping):
            code = payload.get("code") or payload.get("type")
            if code is not None:
                return str(code)
    code = getattr(exc, "code", None)
    return str(code) if code is not None else ""


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not isinstance(headers, Mapping):
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        try:
            moment = parsedate_to_datetime(str(raw)).timestamp()
        except (TypeError, ValueError, OverflowError):
            return None
        return max(0.0, moment - time.time())


def close_stream(stream: Any | None) -> None:
    close = getattr(stream, "close", None)
    if callable(close):
        close()


__all__ = [
    "chunk_fields",
    "close_stream",
    "empty_response_error",
    "empty_stream_error",
    "first_choice",
    "map_provider_error",
    "message_text",
    "output_limit_error",
]
