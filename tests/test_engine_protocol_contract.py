from __future__ import annotations

import json
import re
from pathlib import Path

from soca.app.engine import _model_protocol_payload, _ProtocolWriter, run_engine
from soca.core.workflow.contracts import TerminalStatus, TurnNode
from soca.core.workflow.events import PROTOCOL_VERSION, EventStatus, EventType
from soca.core.workflow.protocol import CURRENT_PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSIONS
from tests.test_engine import ProtocolCapture, make_text_config

ROOT = Path(__file__).resolve().parents[1]
ENGINE_SOURCE = ROOT / "soca" / "app" / "engine.py"
VOICE_SOURCE = ROOT / "soca" / "app" / "voice_controller.py"
PROTOCOL_DOC = ROOT / "docs" / "18-engine-protocol.md"

# Emitted from the workflow envelope, not as flat `"event": "..."` literals, so
# a source scan of engine.py cannot see them.
WORKFLOW_EVENT_NAMES = frozenset(member.value for member in EventType)
SESSION_EVENT_NAMES = frozenset({"session_snapshot", "session_turns_page"})


def _doc() -> str:
    return PROTOCOL_DOC.read_text(encoding="utf-8")


def _doc_cells() -> str:
    """The document with table-cell padding collapsed.

    A markdown formatter pads every cell to align the column, turning
    ``| `done` |`` into ``| `done`    |``. That padding is layout, not content,
    and matching on it made these assertions fail on a pure reformat. Collapsing
    runs of spaces around the pipes keeps the checks about what the table says.
    """
    return re.sub(r"[ \t]*\|[ \t]*", " | ", _doc())


def _engine_source() -> str:
    return ENGINE_SOURCE.read_text(encoding="utf-8")


def _flat_event_names() -> frozenset[str]:
    """Every event name the engine writes as a literal."""
    names = set(re.findall(r'"event":\s*"([a-z_]+)"', _engine_source()))
    # `hello` is built by soca.core.workflow.protocol.protocol_hello.
    names.add("hello")
    # These two frames use the shared `_session_snapshot_payload` constructor,
    # so their event name is a function argument rather than a dict literal.
    names.update(SESSION_EVENT_NAMES)
    return frozenset(names)


def _doc_section(heading: str, next_heading: str) -> str:
    """Slice one `## N. …` section out of the doc.

    Tables in this document are not unique per section — the chat `type` table
    and the command table have the same row shape — so every doc scan has to be
    scoped or it will match rows from a neighbouring section.
    """
    doc = _doc()
    start = doc.index(heading)
    return doc[start : doc.index(next_heading, start)]


def _documented_flat_events() -> frozenset[str]:
    """Event names documented as `### \\`name\\`` sections, plus §5's own heading."""
    documented = set(re.findall(r"^### `([a-z_]+)`$", _doc(), re.MULTILINE))
    # `voice` carries a whole section rather than a `###` entry.
    documented |= set(re.findall(r"^## \d+\. `([a-z_]+)` events$", _doc(), re.MULTILINE))
    return frozenset(documented)


def _dispatch_commands() -> frozenset[str]:
    source = _engine_source()
    body = source[source.index("def dispatch(") : source.index("def shutdown(")]
    return frozenset(re.findall(r'cmd == "([a-z_]+)"', body))


def _documented_commands() -> frozenset[str]:
    """Commands listed in the §2 table body.

    Parsing starts after the `| --- |` separator so the header cell (`cmd`) is
    not mistaken for a command name.
    """
    section = _doc_section("## 2. Commands", "## 3. Event envelope")
    body = re.sub(r"[ \t]*\|[ \t]*", " | ", section)
    # The separator row is matched by shape, not by a literal: a formatter sets
    # each run of dashes to the column width, so `| --- |` becomes `| ------ |`.
    separator = re.search(r"^\s*\|[\s|:-]*-[\s|:-]*\|\s*$", body, re.MULTILINE)
    assert separator is not None, "§2 command table has no separator row"
    body = body[separator.end() :]
    return frozenset(re.findall(r"^ \| `([a-z_]+)` \|", body, re.MULTILINE))


def test_protocol_version_is_pinned() -> None:
    """A version bump must be a deliberate, reviewed change."""
    assert PROTOCOL_VERSION == 3
    assert CURRENT_PROTOCOL_VERSION == PROTOCOL_VERSION
    assert SUPPORTED_PROTOCOL_VERSIONS == (3,)
    assert "| Protocol version | `3`" in _doc_cells()


def test_protocol_writer_preserves_unicode_on_a_legacy_windows_console() -> None:
    """The sidecar protocol must not inherit a Windows console code page."""

    class Cp1252Stream:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def write(self, value: str) -> int:
            value.encode("cp1252")
            self.lines.append(value)
            return len(value)

        def flush(self) -> None:
            return None

    stream = Cp1252Stream()
    _ProtocolWriter(stream).emit({"event": "session_snapshot", "title": "Mở lại phiên"})

    assert json.loads("".join(stream.lines)) == {
        "event": "session_snapshot",
        "title": "Mở lại phiên",
    }


def test_every_emitted_event_is_documented() -> None:
    """The drift gate: an event the engine emits but docs/18 does not describe."""
    undocumented = _flat_event_names() - _documented_flat_events()
    assert not undocumented, (
        f"engine emits undocumented events: {sorted(undocumented)}. "
        "Add a `### `name`` section to docs/18-engine-protocol.md."
    )


def test_every_documented_event_is_still_emitted() -> None:
    """The reverse gate: docs describing an event that no longer exists."""
    stale = _documented_flat_events() - _flat_event_names() - WORKFLOW_EVENT_NAMES
    assert not stale, (
        f"docs/18 documents events the engine never emits: {sorted(stale)}. "
        "Remove the section or restore the emit site."
    )


def test_command_set_matches_documentation() -> None:
    dispatch = _dispatch_commands()
    documented = _documented_commands()
    assert dispatch - documented == set(), (
        f"dispatch accepts undocumented commands: {sorted(dispatch - documented)}"
    )
    assert documented - dispatch == set(), (
        f"docs/18 documents commands dispatch rejects: {sorted(documented - dispatch)}"
    )


def test_command_set_is_pinned() -> None:
    """Removing a command breaks every existing client; make it a visible change."""
    assert _dispatch_commands() == {
        "status",
        "context",
        "memory",
        "memory_compact",
        "memory_proposals",
        "memory_approve",
        "memory_reject",
        "usage",
        "llm_providers",
        "llm_models",
        "llm_set_key",
        "llm_select",
        "llm_config",
        "chat",
        "voice_start",
        "voice_stop",
        "voice_profile_select",
        "knowledge_init",
        "knowledge_model_install",
        "knowledge_index",
        "citation_preview",
        "sessions_list",
        "session_create",
        "session_open",
        "session_turns",
        "session_rename",
        "session_delete",
        "session_status",
        "session_preferences_get",
        "session_preferences_set",
        # `quit` returns before the cmd chain, so it is asserted separately.
    } | {"quit"}


def test_chat_event_types_are_pinned() -> None:
    source = _engine_source()
    chat_types = set(re.findall(r'"event": "chat",\s*\n\s*"type": "([a-z_]+)"', source)) | set(
        re.findall(r'\{"event": "chat", "type": "([a-z_]+)"', source)
    )
    assert chat_types == {"loading", "ready", "start", "done", "error"}
    for chat_type in sorted(chat_types):
        assert f"| `{chat_type}` |" in _doc_cells(), (
            f"chat type {chat_type} missing from docs/18 §4"
        )


def test_voice_event_types_are_pinned() -> None:
    """The 20 VoiceMonitorEvent types a client must switch on."""
    voice_types = frozenset(
        re.findall(r'VoiceMonitorEvent\(\s*"([a-z_]+)"', VOICE_SOURCE.read_text(encoding="utf-8"))
    )
    assert voice_types == {
        "asr_partial",
        "audio",
        "barge_in",
        "done",
        "error",
        "loading",
        "loop_started",
        "loop_stopped",
        "playback_started",
        "progress",
        "ready",
        "recorded",
        "recording",
        "repair",
        "transcribing",
        "tts",
        "turn_end",
        "turn_start",
        "voice_level",
        "warmup",
    }
    doc = _doc()
    for voice_type in sorted(voice_types):
        assert f"`{voice_type}`" in doc, f"voice type {voice_type} missing from docs/18 §5"


def test_workflow_enums_are_pinned() -> None:
    assert {member.value for member in EventType} == {
        "turn_started",
        "step_started",
        "step_progress",
        "step_completed",
        "verification_started",
        "verification_completed",
        "answer_delta",
        "public_update",
        "goal_resolved",
        "turn_terminal",
    }
    assert {member.value for member in EventStatus} == {
        "started",
        "active",
        "completed",
        "failed",
        "cancelled",
    }
    assert {member.value for member in TerminalStatus} == {
        "achieved",
        "needs_clarification",
        "insufficient_evidence",
        "safe_failure",
        "budget_exhausted",
        "cancelled",
        "system_failure",
    }
    assert len({member.value for member in TurnNode}) == 13


def test_workflow_vocabulary_is_documented() -> None:
    doc = _doc()
    for enum in (EventType, EventStatus, TerminalStatus, TurnNode):
        for member in enum:
            assert f"`{member.value}`" in doc, (
                f"{enum.__name__}.{member.name} ({member.value}) missing from docs/18 §6"
            )


def test_llm_catalog_model_fields_are_pinned() -> None:
    """`supported_parameters` must stay out of the protocol (docs/18 §4)."""

    class _Model:
        id = "openai/gpt-5.6-luna"
        label = "GPT-5.6 Luna"
        context_length = 200_000
        price_prompt_per_1m = 1.0
        price_completion_per_1m = 2.0
        pricing_source = "api"
        max_output_tokens = 8_192
        reasoning_supported = True
        reasoning_mandatory = False
        supported_parameters = ("temperature", "top_p")

    payload = _model_protocol_payload(_Model())
    assert set(payload) == {
        "id",
        "label",
        "context_length",
        "price_prompt_per_1m",
        "price_completion_per_1m",
        "pricing_source",
        "max_output_tokens",
        "reasoning_supported",
        "reasoning_mandatory",
    }
    assert "supported_parameters" not in payload


def test_hello_envelope_matches_documentation() -> None:
    """Live run: the first frame must carry the version fields a client checks."""
    capture = ProtocolCapture()

    def stdin():
        yield '{"cmd": "quit"}\n'

    code = run_engine(
        voice_config=None,
        text_config=make_text_config(),
        profile="baseline",
        no_model=True,
        stdin=stdin(),
        stdout=capture,
    )

    assert code == 0
    events = capture.events()
    hello = events[0]
    assert hello["event"] == "hello"
    assert set(hello) >= {
        "event",
        "version",
        "protocol_version",
        "supported_versions",
        "profile",
        "no_model",
        "stack",
    }
    assert hello["protocol_version"] == PROTOCOL_VERSION
    assert hello["supported_versions"] == list(SUPPORTED_PROTOCOL_VERSIONS)
    # docs/18 §1: hello is always followed by context, and bye is the last frame.
    assert events[1]["event"] == "context"
    assert events[-1] == {"event": "bye"}


def test_unknown_command_does_not_terminate_the_engine() -> None:
    """docs/18 §1: malformed input is reported, never fatal."""
    capture = ProtocolCapture()

    def stdin():
        yield "not json at all\n"
        yield "[1, 2, 3]\n"
        yield '{"cmd": "nope"}\n'
        yield '{"cmd": "quit"}\n'

    code = run_engine(
        voice_config=None,
        text_config=make_text_config(),
        profile="baseline",
        no_model=True,
        stdin=stdin(),
        stdout=capture,
    )

    assert code == 0
    events = capture.events()
    errors = [event for event in events if event["event"] == "engine_error"]
    assert len(errors) == 3
    assert errors[0]["message"].startswith("invalid JSON")
    assert errors[1]["message"] == "command must be a JSON object"
    assert errors[2]["message"] == "unknown command: 'nope'"
    assert events[-1] == {"event": "bye"}


def test_every_frame_is_one_json_object_per_line() -> None:
    """NDJSON framing: no frame may contain a raw newline."""
    capture = ProtocolCapture()

    def stdin():
        yield '{"cmd": "status"}\n'
        yield '{"cmd": "usage"}\n'
        yield '{"cmd": "quit"}\n'

    run_engine(
        voice_config=None,
        text_config=make_text_config(),
        profile="baseline",
        no_model=True,
        stdin=stdin(),
        stdout=capture,
    )

    for line in capture.lines:
        assert "\n" not in line
        payload = json.loads(line)
        assert isinstance(payload, dict)
        assert isinstance(payload.get("event"), str)
