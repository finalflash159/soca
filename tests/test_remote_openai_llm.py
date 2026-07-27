from __future__ import annotations

from types import SimpleNamespace

import pytest

from soca.llm.providers.provider_registry import get_provider
from soca.llm.providers.remote_openai_llm import RemoteLLMError, RemoteOpenAILLM

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
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))], usage=None)


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
        if self._owner.raises is not None:
            raise self._owner.raises
        if kwargs.get("stream"):
            return iter(self._owner.stream_chunks)
        return self._owner.completion


class FakeClient:
    def __init__(self, *, completion=None, stream_chunks=None, raises=None) -> None:
        self.completion = completion
        self.stream_chunks = stream_chunks or []
        self.raises = raises
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=FakeChatCompletions(self))


class FakeStatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class FakeConnectionError(Exception):
    """Mimics openai.APIConnectionError (no status_code)."""


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


def _engine(client: FakeClient, *, model: str = "llama-3.1-8b-instant", provider_key: str = "groq"):
    return RemoteOpenAILLM(
        provider=get_provider(provider_key),
        model=model,
        api_key="sk-test-key",
        client=client,
    )


def test_generate_maps_response_and_usage_into_result():
    client = FakeClient(completion=_make_completion("Chào bạn.", prompt_tokens=25, completion_tokens=4))
    engine = _engine(client)

    result = engine.generate("Xin chào", max_tokens=64)

    assert result.text == "Chào bạn."
    assert result.n_prompt_tokens == 25
    assert result.n_completion_tokens == 4
    assert result.total_latency_ms >= 0
    call = client.calls[0]
    assert call["model"] == "llama-3.1-8b-instant"
    assert call["max_tokens"] == 64
    assert call["stream"] is False


def test_generate_injects_soca_persona_system_message():
    client = FakeClient(completion=_make_completion("ok", 10, 1))
    engine = _engine(client)

    engine.generate("Thời tiết hôm nay?", inject_persona=True)

    messages = client.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "Sơn Ca" in messages[0]["content"]
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"] == "Thời tiết hôm nay?"


def test_generate_without_persona_has_no_forced_system_prompt():
    client = FakeClient(completion=_make_completion("ok", 10, 1))
    engine = _engine(client)

    engine.generate("Câu hỏi trần", inject_persona=False)

    roles = [m["role"] for m in client.calls[0]["messages"]]
    assert "system" not in roles


def test_generate_strips_whitespace_from_completion():
    client = FakeClient(completion=_make_completion("  Trả lời.\n", 5, 2))
    engine = _engine(client)
    assert engine.generate("hi").text == "Trả lời."


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

    with pytest.raises(RemoteLLMError, match="max_tokens"):
        engine.generate("hi", max_tokens=40)


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


def test_generate_stream_validates_arguments():
    engine = _engine(FakeClient(stream_chunks=[]))
    with pytest.raises(ValueError):
        list(engine.generate_stream(""))


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


def test_rate_limit_error_maps_to_friendly_message():
    engine = _engine(FakeClient(raises=FakeStatusError(429)))
    with pytest.raises(RemoteLLMError) as excinfo:
        engine.generate("hi")
    assert excinfo.value.category == "rate_limit"


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
