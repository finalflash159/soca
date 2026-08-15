from __future__ import annotations

from collections.abc import Iterator

import pytest
from rich.console import Console

from soca.app.text_runtime import render_text_result, stream_text_answer
from soca.core.turn import RuntimeResult, RuntimeRoute, RuntimeStreamEvent


class _StreamingRuntime:
    CHUNKS = ("Protein giữ cơ bắp [K1]. ", "Nó tạo cảm giác no [K2].")

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def stream_text_turn(
        self, text: str, *, source: str, metadata: dict, **kwargs
    ) -> Iterator[RuntimeStreamEvent]:
        del kwargs
        self.calls.append({"text": text, "source": source, "metadata": metadata})
        for chunk in self.CHUNKS:
            yield RuntimeStreamEvent(type="sentence", text=chunk)
        yield RuntimeStreamEvent(
            type="result",
            result=RuntimeResult(
                response_text="".join(self.CHUNKS),
                route=RuntimeRoute.FREE_CHAT,
            ),
        )


class _NoResultRuntime:
    def stream_text_turn(self, text: str, *, source: str, metadata: dict, **kwargs):
        del text, source, metadata, kwargs
        yield RuntimeStreamEvent(type="sentence", text="một nửa")


class _BlockingOnlyRuntime:
    def run_text_turn(self, text: str, *, source: str, metadata: dict) -> RuntimeResult:
        del text, source, metadata
        return RuntimeResult(response_text="không stream", route=RuntimeRoute.FREE_CHAT)


def _console() -> Console:
    return Console(record=True, width=200, no_color=True, force_terminal=False)


def test_cli_prints_each_chunk_and_returns_the_result() -> None:
    console = _console()
    runtime = _StreamingRuntime()

    result = stream_text_answer(console, runtime, "chất đạm", source="cli", metadata={})

    assert result.response_text == "".join(_StreamingRuntime.CHUNKS)
    assert runtime.calls[0]["source"] == "cli"
    printed = console.export_text()
    assert "Protein giữ cơ bắp. Nó tạo cảm giác no." in printed
    assert "[K1]" not in printed, "a terminal must not show labels the citations table renders"


def test_cli_does_not_print_the_answer_twice() -> None:
    console = _console()
    runtime = _StreamingRuntime()

    result = stream_text_answer(console, runtime, "chất đạm", source="cli", metadata={})
    render_text_result(console, result, answer_printed=True)

    assert console.export_text().count("Protein giữ cơ bắp") == 1


def test_cli_renders_the_answer_when_it_was_not_streamed() -> None:
    console = _console()
    result = RuntimeResult(response_text="câu trả lời", route=RuntimeRoute.FREE_CHAT)

    render_text_result(console, result)

    assert "câu trả lời" in console.export_text()


def test_cli_refuses_a_runtime_that_cannot_stream() -> None:
    with pytest.raises(TypeError, match="stream_text_turn"):
        stream_text_answer(
            _console(), _BlockingOnlyRuntime(), "chất đạm", source="cli", metadata={}
        )


def test_cli_refuses_a_stream_that_never_produced_a_result() -> None:
    with pytest.raises(RuntimeError, match="result event"):
        stream_text_answer(_console(), _NoResultRuntime(), "chất đạm", source="cli", metadata={})
