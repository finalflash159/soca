from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("textual")

from textual.geometry import Offset
from textual.selection import Selection
from textual.widgets import Input

from soca.app.tui import SoCaTuiApp, TuiConfig
from soca.app.tui.widgets import (
    InspectorWidget,
    SidebarWidget,
    SlashCommandListWidget,
    TimelineWidget,
)


def test_status_tui_does_not_build_runtime() -> None:
    called = False

    def runtime_builder(_config):
        nonlocal called
        called = True
        raise AssertionError("status mode must not build a text runtime")

    async def run() -> None:
        app = SoCaTuiApp(
            TuiConfig(mode="status", no_model=True, show_splash=False),
            runtime_builder=runtime_builder,
        )
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert called is False
            assert app.bundle is None
            inspector = app.query_one("#inspector", InspectorWidget)
            assert inspector is not None
            composer = app.query_one("#composer", Input)
            assert "Status mode" in composer.placeholder

    asyncio.run(run())


def test_status_tui_plain_text_does_not_build_runtime() -> None:
    called = False

    def runtime_builder(_config):
        nonlocal called
        called = True
        raise AssertionError("status mode must not send plain text to runtime")

    async def run() -> None:
        app = SoCaTuiApp(
            TuiConfig(mode="status", no_model=True, show_splash=False),
            runtime_builder=runtime_builder,
        )
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            composer = app.query_one("#composer", Input)
            composer.value = "xin chào"
            await pilot.press("enter")
            await pilot.pause()

            assert called is False
            assert app.state.mode == "status"

    asyncio.run(run())


def test_status_tui_slash_chat_switches_mode() -> None:
    async def run() -> None:
        app = SoCaTuiApp(
            TuiConfig(mode="status", no_model=True, show_splash=False),
            runtime_builder=lambda _config: (_ for _ in ()).throw(AssertionError("not used")),
        )
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            composer = app.query_one("#composer", Input)
            composer.value = "/chat"
            await pilot.press("enter")
            await pilot.pause()

            assert app.state.mode == "chat"
            assert "Chat với SoCa" in composer.placeholder

    asyncio.run(run())


def test_sidebar_reflects_active_mode() -> None:
    async def run() -> None:
        app = SoCaTuiApp(
            TuiConfig(mode="status", no_model=True, show_splash=False),
            runtime_builder=lambda _config: (_ for _ in ()).throw(AssertionError("not used")),
        )
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            sidebar = app.query_one("#sidebar", SidebarWidget)
            assert sidebar.active == "status"

            composer = app.query_one("#composer", Input)
            composer.value = "/chat"
            await pilot.press("enter")
            await pilot.pause()

            assert sidebar.active == "chat"

    asyncio.run(run())


def test_bare_q_is_not_an_exit_command() -> None:
    async def run() -> None:
        app = SoCaTuiApp(
            TuiConfig(mode="status", no_model=True, show_splash=False),
            runtime_builder=lambda _config: (_ for _ in ()).throw(AssertionError("not used")),
        )
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            composer = app.query_one("#composer", Input)
            composer.value = "q"
            await pilot.press("enter")
            await pilot.pause()

            # App is still alive (composer queryable) and did not exit on "q".
            assert app.query_one("#composer", Input) is not None
            assert app.state.mode == "status"

    asyncio.run(run())


def test_slash_suggestions_filter_while_typing() -> None:
    async def run() -> None:
        app = SoCaTuiApp(TuiConfig(mode="status", no_model=True, show_splash=False))
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            composer = app.query_one("#composer", Input)
            suggestions = app.query_one("#slash_commands", SlashCommandListWidget)

            composer.value = "/m"
            await pilot.pause()

            assert suggestions.display is True
            assert "/memory" in suggestions.last_text

            composer.value = "xin chào"
            await pilot.pause()

            assert suggestions.display is False

    asyncio.run(run())


def test_tui_keeps_ctrl_c_available_for_textual_copy() -> None:
    assert all(binding[0] != "ctrl+c" for binding in SoCaTuiApp.BINDINGS)


def test_tui_selection_style_is_visible() -> None:
    styles_path = Path("soca/app/tui/styles.tcss")
    styles = styles_path.read_text(encoding="utf-8")

    assert ".screen--selection" in styles
    assert "background: #1f6feb 70%" in styles


def test_timeline_widget_supports_textual_selection() -> None:
    timeline = TimelineWidget(wrap=True, highlight=False, markup=False)
    timeline.add_user("xin chào")
    timeline.add_assistant("Chào bạn.")

    text, ending = timeline.get_selection(
        Selection.from_offsets(Offset(0, 0), Offset(80, 2))
    )

    assert text.startswith("You ▸")
    assert "SoCa" in text
    assert ending == "\n"


def test_copy_command_copies_plain_timeline_text() -> None:
    async def run() -> None:
        app = SoCaTuiApp(TuiConfig(mode="status", no_model=True, show_splash=False))
        copied: list[str] = []
        app.copy_to_clipboard = copied.append  # type: ignore[method-assign]

        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            app.timeline.add_user("xin chào")
            app.timeline.add_assistant("Chào bạn.")
            composer = app.query_one("#composer", Input)
            composer.value = "/copy"
            await pilot.press("enter")
            await pilot.pause()

            assert len(copied) == 1
            assert "You ▸ xin chào" in copied[0]
            assert "(o> SoCa ▸ Chào bạn." in copied[0]

    asyncio.run(run())
