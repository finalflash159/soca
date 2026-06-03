from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

pytest.importorskip("textual")

from textual.widgets import Input

from soca.app.text_runtime import TextRuntimeBundle, TextRuntimeConfig
from soca.app.tui import SoCaTuiApp, TuiConfig
from soca.core.turn import RuntimeResult, RuntimeRoute, RuntimeTrace
from soca.core.usage import LLMUsage
from soca.memory import SessionMemory


@dataclass
class FakeRuntime:
    calls: list[str]

    def run_text_turn(self, text: str, *, source: str, metadata: dict[str, object]) -> RuntimeResult:
        self.calls.append(text)
        return RuntimeResult(
            response_text=f"Đã nhận: {text}",
            route=RuntimeRoute.FREE_CHAT,
            trace=RuntimeTrace(
                route=RuntimeRoute.FREE_CHAT,
                used_llm=True,
                stage_latencies_ms={"llm": 12.0},
            ),
            usage=LLMUsage(
                prompt_tokens=8,
                completion_tokens=4,
                ttft_ms=2.0,
                total_latency_ms=12.0,
                tokens_per_second=50.0,
            ),
        )


def test_chat_tui_uses_injected_runtime_and_tracks_usage() -> None:
    fake_runtime = FakeRuntime(calls=[])

    def runtime_builder(_config: TextRuntimeConfig) -> TextRuntimeBundle:
        return TextRuntimeBundle(
            runtime=fake_runtime,  # type: ignore[arg-type]
            session_memory=SessionMemory(),
            llm_status="enabled:fake",
            knowledge_status="enabled:fake",
            memory_status="enabled:fake",
        )

    async def run() -> None:
        app = SoCaTuiApp(
            TuiConfig(mode="chat", no_model=True, show_splash=False),
            runtime_builder=runtime_builder,
        )
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            composer = app.query_one("#composer", Input)
            composer.value = "xin chào"
            await pilot.press("enter")
            await pilot.pause()

            assert fake_runtime.calls == ["xin chào"]
            assert app.state.session_usage.total_turns == 1
            assert app.state.session_usage.total_prompt_tokens == 8
            assert app.last_result is not None

    asyncio.run(run())


def test_chat_busy_guard_blocks_concurrent_submit() -> None:
    fake_runtime = FakeRuntime(calls=[])

    def runtime_builder(_config: TextRuntimeConfig) -> TextRuntimeBundle:
        return TextRuntimeBundle(
            runtime=fake_runtime,  # type: ignore[arg-type]
            session_memory=SessionMemory(),
            llm_status="enabled:fake",
            knowledge_status="enabled:fake",
            memory_status="enabled:fake",
        )

    async def run() -> None:
        app = SoCaTuiApp(
            TuiConfig(mode="chat", no_model=True, show_splash=False),
            runtime_builder=runtime_builder,
        )
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app._chat_busy = True  # simulate a turn already in flight
            composer = app.query_one("#composer", Input)
            composer.value = "xin chào"
            await pilot.press("enter")
            await pilot.pause()

            # Guard must prevent a concurrent run_text_turn on the shared runtime.
            assert fake_runtime.calls == []

    asyncio.run(run())


def test_tui_chat_runtime_receives_shared_session_memory() -> None:
    captured: dict[str, SessionMemory | None] = {}

    def runtime_builder(
        _config: TextRuntimeConfig,
        *,
        session_memory: SessionMemory | None = None,
    ) -> TextRuntimeBundle:
        captured["session"] = session_memory
        return TextRuntimeBundle(
            runtime=FakeRuntime(calls=[]),  # type: ignore[arg-type]
            session_memory=session_memory,
            llm_status="enabled:fake",
            knowledge_status="enabled:fake",
            memory_status="enabled:fake",
        )

    async def run() -> None:
        app = SoCaTuiApp(
            TuiConfig(mode="chat", no_model=True, show_splash=False),
            runtime_builder=runtime_builder,
        )
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            await app._ensure_bundle()

            assert captured["session"] is app.shared_session_memory
            assert app.bundle is not None
            assert app.bundle.session_memory is app.shared_session_memory

    asyncio.run(run())
