from __future__ import annotations

from types import SimpleNamespace

import pytest

from soca.llm.providers.provider_registry import get_provider
from soca.llm.providers.remote_openai_llm import RemoteLLMError, RemoteOpenAILLM
from soca.llm.providers.response_adapter import map_provider_error

# ---------------------------------------------------------------------------
# Fake OpenAI client (no network). Mirrors the shape of openai>=1.x responses.
# ---------------------------------------------------------------------------


def _make_completion(text: str, prompt_tokens: int, completion_tokens: int):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


def _make_delta(content: str | None):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content))], usage=None
    )


def _make_usage_only(prompt_tokens: int, completion_tokens: int):
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


class FakeChatCompletions:
    def __init__(self, owner: FakeClient) -> None:
        self._owner = owner

    def create(self, **kwargs):
        self._owner.calls.append(kwargs)
        if self._owner.raise_sequence:
            error = self._owner.raise_sequence.pop(0)
            if error is not None:
                raise error
        if self._owner.raises is not None:
            raise self._owner.raises
        if kwargs.get("stream"):
            if self._owner.stream_sequence:
                return iter(self._owner.stream_sequence.pop(0))
            return iter(self._owner.stream_chunks)
        return self._owner.completion


class FakeClient:
    def __init__(
        self,
        *,
        completion=None,
        stream_chunks=None,
        stream_sequence=None,
        raises=None,
        raise_sequence=None,
    ) -> None:
        self.completion = completion
        self.stream_chunks = stream_chunks or []
        self.stream_sequence = list(stream_sequence or [])
        self.raises = raises
        self.raise_sequence = list(raise_sequence or [])
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=FakeChatCompletions(self))


class FakeStatusError(Exception):
    def __init__(
        self,
        status_code: int,
        *,
        headers: dict[str, str] | None = None,
        body: dict | None = None,
    ) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code
        self.response = SimpleNamespace(headers=headers or {})
        self.body = body


class FakeConnectionError(ConnectionError):
    """Mimics openai.APIConnectionError (no status_code)."""


class FakeProgrammingError(Exception):
    pass


class SDKInternalError(Exception):
    pass


SDKInternalError.__module__ = "openai"


class ClosableChunks:
    def __init__(self, chunks) -> None:
        self._chunks = iter(chunks)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._chunks)

    def close(self) -> None:
        self.closed = True


class RaisingChunks:
    def __init__(self, chunks, error: Exception) -> None:
        self._chunks = iter(chunks)
        self._error = error

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise self._error from None


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


def _engine(
    client: FakeClient,
    *,
    model: str = "llama-3.1-8b-instant",
    provider_key: str = "groq",
    reasoning_enabled: bool | None = None,
    reasoning_parameter: str | None = None,
):
    return RemoteOpenAILLM(
        provider=get_provider(provider_key),
        model=model,
        api_key="sk-test-key",
        client=client,
        reasoning_enabled=reasoning_enabled,
        reasoning_parameter=reasoning_parameter,
    )


def test_generate_maps_response_and_usage_into_result():
    client = FakeClient(
        completion=_make_completion("Chào bạn.", prompt_tokens=25, completion_tokens=4)
    )
    engine = _engine(client)

    result = engine.generate("Xin chào", max_tokens=64)

    assert result.text == "Chào bạn."
    assert result.n_prompt_tokens == 25
    assert result.n_completion_tokens == 4
    assert result.total_latency_ms >= 0
    call = client.calls[0]
    assert call["model"] == "llama-3.1-8b-instant"
    assert call["max_completion_tokens"] == 64
    assert call["stream"] is False


def test_generate_injects_soca_persona_system_message():
    client = FakeClient(completion=_make_completion("ok", 10, 1))
    engine = _engine(client)

    engine.generate("Thời tiết hôm nay?", inject_persona=True)

    messages = client.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "SoCa" in messages[0]["content"]
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "Thời tiết hôm nay?"


def test_generate_without_persona_has_no_forced_system_prompt():
    client = FakeClient(completion=_make_completion("ok", 10, 1))
    engine = _engine(client)

    engine.generate("Câu hỏi trần", inject_persona=False)

    roles = [m["role"] for m in client.calls[0]["messages"]]
    assert "system" not in roles


def test_unknown_reasoning_capability_omits_control_for_mandatory_models():
    client = FakeClient(completion=_make_completion("ok", 10, 1))
    engine = _engine(client, provider_key="openrouter")

    engine.generate("Câu hỏi")

    assert client.calls[0]["extra_body"] == {"provider": {"allow_fallbacks": False}}


def test_reasoning_capability_controls_any_unified_provider():
    client = FakeClient(completion=_make_completion("ok", 10, 1))
    engine = _engine(
        client,
        provider_key="openrouter",
        reasoning_enabled=True,
        reasoning_parameter="reasoning",
    )

    engine.generate("Câu hỏi")

    assert client.calls[0]["extra_body"] == {
        "provider": {"allow_fallbacks": False},
        "reasoning": {"enabled": True, "exclude": True},
    }


def test_verified_optional_reasoning_can_be_disabled():
    client = FakeClient(completion=_make_completion("ok", 10, 1))
    engine = _engine(
        client,
        provider_key="openrouter",
        reasoning_enabled=False,
        reasoning_parameter="reasoning",
    )

    engine.generate("Câu hỏi")

    assert client.calls[0]["extra_body"] == {
        "provider": {"allow_fallbacks": False},
        "reasoning": {"effort": "none"},
    }


def test_generate_strips_whitespace_from_completion():
    client = FakeClient(completion=_make_completion("  Trả lời.\n", 5, 2))
    engine = _engine(client)
    assert engine.generate("hi").text == "Trả lời."


def test_provider_adapter_clamps_output_and_records_effective_value():
    client = FakeClient(completion=_make_completion("ok", 3, 1))
    engine = RemoteOpenAILLM(
        provider=get_provider("openrouter"),
        model="some/model",
        api_key="sk-test",
        client=client,
        max_output_tokens=2_048,
    )

    result = engine.generate("hi", max_tokens=8_192)

    assert client.calls[0]["max_tokens"] == 2_048
    assert result.provider_trace["requested_max_tokens"] == 8_192
    assert result.provider_trace["effective_max_tokens"] == 2_048


def test_generate_rejects_empty_completion_with_actionable_error():
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=""),
                finish_reason="length",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=40),
    )
    engine = _engine(FakeClient(completion=completion))

    with pytest.raises(RemoteLLMError) as excinfo:
        engine.generate("hi", max_tokens=40)
    assert excinfo.value.category == "output_limit"
    assert engine.last_call_trace is not None
    assert engine.last_call_trace.attempts[-1].failure_kind == "output_limit"


def test_nonempty_truncated_completion_is_not_reported_as_success():
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Câu trả lời còn dang dở"),
                finish_reason="length",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=40),
    )
    engine = _engine(FakeClient(completion=completion))

    with pytest.raises(RemoteLLMError) as excinfo:
        engine.generate("hi", max_tokens=40)

    assert excinfo.value.category == "output_limit"


def test_generate_preserves_provider_refusal():
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="", refusal="Không thể hỗ trợ yêu cầu này."),
                finish_reason="stop",
            )
        ],
        usage=None,
    )
    engine = _engine(FakeClient(completion=completion))

    with pytest.raises(RemoteLLMError) as excinfo:
        engine.generate("hi")

    assert excinfo.value.category == "refusal"
    assert "Không thể hỗ trợ" in str(excinfo.value)


def test_generate_distinguishes_reasoning_only_response():
    completion = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="", reasoning="internal summary"),
                finish_reason="length",
            )
        ],
        usage=None,
    )
    engine = _engine(FakeClient(completion=completion))

    with pytest.raises(RemoteLLMError) as excinfo:
        engine.generate("hi")

    assert excinfo.value.category == "reasoning_only"


def test_generate_rejects_empty_user_message():
    engine = _engine(FakeClient(completion=_make_completion("x", 1, 1)))
    with pytest.raises(ValueError):
        engine.generate("   ")


def test_generate_rejects_non_positive_max_tokens():
    engine = _engine(FakeClient(completion=_make_completion("x", 1, 1)))
    with pytest.raises(ValueError):
        engine.generate("hi", max_tokens=0)


# ---------------------------------------------------------------------------
# generate_stream()
# ---------------------------------------------------------------------------


def test_generate_stream_yields_delta_text_and_requests_usage():
    chunks = [
        _make_delta("Chào "),
        _make_delta("bạn"),
        _make_delta(None),  # tool/keepalive delta with no content
        _make_usage_only(10, 2),
    ]
    client = FakeClient(stream_chunks=chunks)
    engine = _engine(client)

    out = list(engine.generate_stream("Xin chào"))

    assert "".join(out) == "Chào bạn"
    call = client.calls[0]
    assert call["stream"] is True
    assert call["stream_options"] == {"include_usage": True}


def test_stream_uses_model_default_when_capability_is_unknown():
    client = FakeClient(stream_chunks=[_make_delta("ok")])
    engine = _engine(client, provider_key="openrouter")

    list(engine.generate_stream("Câu hỏi"))

    assert client.calls[0]["extra_body"] == {"provider": {"allow_fallbacks": False}}


def test_generate_stream_validates_arguments():
    engine = _engine(FakeClient(stream_chunks=[]))
    with pytest.raises(ValueError):
        list(engine.generate_stream(""))


def test_closing_consumer_stream_cancels_and_closes_provider_stream():
    chunks = ClosableChunks([_make_delta("một"), _make_delta(" hai")])
    engine = _engine(FakeClient(stream_chunks=chunks))

    output = engine.generate_stream("hi")
    assert next(output) == "một"
    output.close()

    assert chunks.closed is True


def test_stream_retries_transient_failure_before_first_output():
    client = FakeClient(
        stream_sequence=[
            RaisingChunks([], FakeStatusError(503)),
            [_make_delta("ok")],
        ]
    )
    delays: list[float] = []
    engine = RemoteOpenAILLM(
        provider=get_provider("groq"),
        model="llama-3.1-8b-instant",
        api_key="sk-test",
        client=client,
        sleep=delays.append,
    )

    assert list(engine.generate_stream("hi")) == ["ok"]
    assert len(client.calls) == 2
    assert delays == [0.25]
    assert engine.last_call_trace is not None
    assert engine.last_call_trace.as_dict()["retry_count"] == 1


def test_stream_does_not_retry_after_emitting_partial_output():
    client = FakeClient(
        stream_chunks=RaisingChunks([_make_delta("partial")], FakeStatusError(503))
    )
    engine = _engine(client)

    stream = engine.generate_stream("hi")
    assert next(stream) == "partial"
    with pytest.raises(RemoteLLMError) as excinfo:
        next(stream)

    assert excinfo.value.retryable is False
    assert len(client.calls) == 1
    assert engine.last_call_trace is not None
    assert engine.last_call_trace.attempts[-1].outcome == "failure"


def test_empty_stream_has_typed_reason_instead_of_silent_success():
    engine = _engine(FakeClient(stream_chunks=[]))

    with pytest.raises(RemoteLLMError) as excinfo:
        list(engine.generate_stream("hi"))

    assert excinfo.value.category == "empty_response"


def test_stream_finish_length_is_terminal_even_after_partial_text():
    finish = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason="length")]
    )
    engine = _engine(FakeClient(stream_chunks=[_make_delta("partial"), finish]))

    stream = engine.generate_stream("hi")
    assert next(stream) == "partial"
    with pytest.raises(RemoteLLMError) as excinfo:
        next(stream)

    assert excinfo.value.category == "output_limit"
    assert excinfo.value.retryable is False
    # The generic "stream was interrupted" wording used to replace the specific
    # cause, so the one actionable fact — raise the output budget — was lost.
    assert "ngân sách output" in str(excinfo.value)


# ---------------------------------------------------------------------------
# count_tokens()
# ---------------------------------------------------------------------------


def test_count_tokens_zero_for_empty():
    engine = _engine(FakeClient(completion=_make_completion("x", 1, 1)))
    assert engine.count_tokens("") == 0


def test_count_tokens_positive_for_text():
    engine = _engine(FakeClient(completion=_make_completion("x", 1, 1)))
    assert engine.count_tokens("một câu tiếng Việt dài") > 0


# ---------------------------------------------------------------------------
# error mapping
# ---------------------------------------------------------------------------


def test_auth_error_maps_to_friendly_message():
    engine = _engine(FakeClient(raises=FakeStatusError(401)))
    with pytest.raises(RemoteLLMError) as excinfo:
        engine.generate("hi")
    assert excinfo.value.category == "auth"
    assert "key" in str(excinfo.value).lower()
    assert len(engine._client.calls) == 1


def test_rate_limit_error_maps_to_friendly_message():
    engine = _engine(FakeClient(raises=FakeStatusError(429)))
    with pytest.raises(RemoteLLMError) as excinfo:
        engine.generate("hi")
    assert excinfo.value.category == "rate_limit"


def test_transient_error_retries_in_one_observable_ledger():
    client = FakeClient(
        completion=_make_completion("ok", 2, 1),
        raise_sequence=[FakeStatusError(503), None],
    )
    delays: list[float] = []
    engine = RemoteOpenAILLM(
        provider=get_provider("groq"),
        model="llama-3.1-8b-instant",
        api_key="sk-test",
        client=client,
        sleep=delays.append,
    )

    result = engine.generate("hi")

    assert result.text == "ok"
    assert len(client.calls) == 2
    assert delays == [0.25]
    assert result.provider_trace["retry_count"] == 1
    assert result.provider_trace["attempts"][0]["failure_kind"] == "server"


def test_retry_after_and_provider_error_code_are_preserved():
    client = FakeClient(
        completion=_make_completion("ok", 2, 1),
        raise_sequence=[
            FakeStatusError(
                429,
                headers={"Retry-After": "1.5"},
                body={"error": {"code": "quota_exceeded"}},
            ),
            None,
        ],
    )
    delays: list[float] = []
    engine = RemoteOpenAILLM(
        provider=get_provider("openrouter"),
        model="some/model",
        api_key="sk-test",
        client=client,
        sleep=delays.append,
    )

    engine.generate("hi")

    assert delays == [1.5]
    assert engine.last_call_trace is not None
    assert engine.last_call_trace.attempts[0].failure_kind == "rate_limit"

    failing = RemoteOpenAILLM(
        provider=get_provider("openrouter"),
        model="some/model",
        api_key="sk-test",
        client=FakeClient(
            raises=FakeStatusError(
                429,
                headers={"Retry-After": "1.5"},
                body={"error": {"code": "quota_exceeded"}},
            )
        ),
        max_retries=0,
    )
    with pytest.raises(RemoteLLMError) as excinfo:
        failing.generate("hi")
    assert excinfo.value.provider_code == "quota_exceeded"
    assert excinfo.value.retry_after_s == 1.5


def test_programming_exception_is_not_translated_or_retried():
    engine = _engine(FakeClient(raises=FakeProgrammingError("bug")))

    with pytest.raises(FakeProgrammingError, match="bug"):
        engine.generate("hi")

    assert len(engine._client.calls) == 1


def test_sdk_module_exception_is_not_assumed_to_be_a_transport_failure():
    error = SDKInternalError("sdk bug")
    assert map_provider_error(error, get_provider("groq"), "test/model") is None
    engine = _engine(FakeClient(raises=error))

    with pytest.raises(SDKInternalError, match="sdk bug"):
        engine.generate("hi")

    assert len(engine._client.calls) == 1


def test_connection_error_maps_to_friendly_message():
    engine = _engine(FakeClient(raises=FakeConnectionError("no route")))
    with pytest.raises(RemoteLLMError) as excinfo:
        engine.generate("hi")
    assert excinfo.value.category == "network"
    assert "groq" in str(excinfo.value).lower()


def test_stream_errors_are_also_mapped():
    engine = _engine(FakeClient(raises=FakeStatusError(401)))
    with pytest.raises(RemoteLLMError):
        list(engine.generate_stream("hi"))
