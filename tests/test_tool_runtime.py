from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from soca.knowledge import MarkdownVaultKnowledgeSource
from soca.tools import (
    KnowledgeReadTool,
    KnowledgeSearchTool,
    SideEffectLevel,
    ToolCall,
    ToolExecutionStatus,
    ToolResult,
    ToolRuntime,
    ToolSpec,
    object_schema,
)


@dataclass
class EchoTool:
    name: str = "echo"
    enabled: bool = True
    side_effect: SideEffectLevel = SideEffectLevel.READ_ONLY

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description="Echo a message.",
            input_schema=object_schema(
                properties={"message": {"type": "string"}},
                required=["message"],
            ),
            side_effect=self.side_effect,
            enabled=self.enabled,
        )

    def run(self, arguments: dict) -> ToolResult:
        return ToolResult(
            name=self.spec.name,
            ok=True,
            content=str(arguments["message"]),
            data={"message": arguments["message"]},
        )


def make_vault(root: Path) -> MarkdownVaultKnowledgeSource:
    wiki = root / "wiki"
    wiki.mkdir()
    (wiki / "nutrition.md").write_text(
        "# Dinh dưỡng\nNên ăn đủ đạm, rau xanh và uống đủ nước.\n#dinh-duong",
        encoding="utf-8",
    )
    (root / "private").mkdir()
    (root / "private" / "secret.md").write_text("# Secret", encoding="utf-8")
    return MarkdownVaultKnowledgeSource(root, include_globs=("wiki/**/*.md",))


def test_tool_runtime_calls_registered_tool() -> None:
    runtime = ToolRuntime([EchoTool()])

    result = runtime.call(ToolCall("echo", {"message": "xin chào"}))

    assert result.ok is True
    assert result.content == "xin chào"
    assert result.data == {"message": "xin chào"}


def test_tool_runtime_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="Duplicate tool name"):
        ToolRuntime([EchoTool(), EchoTool()])


def test_tool_runtime_lists_specs_in_deterministic_order() -> None:
    runtime = ToolRuntime([EchoTool(name="z.tool"), EchoTool(name="a.tool")])

    assert [spec.name for spec in runtime.list_specs()] == ["a.tool", "z.tool"]


def test_tool_runtime_validates_required_and_types() -> None:
    runtime = ToolRuntime([EchoTool()])

    missing = runtime.call(ToolCall("echo", {}))
    wrong_type = runtime.call(ToolCall("echo", {"message": 123}))

    assert missing.ok is False
    assert missing.status is ToolExecutionStatus.INVALID
    assert missing.error == "Missing required argument: message"
    assert wrong_type.ok is False
    assert wrong_type.error == "Argument message must be string; got integer"


def test_tool_runtime_rejects_disabled_and_excessive_side_effects() -> None:
    disabled_runtime = ToolRuntime([EchoTool(enabled=False)])
    restricted_runtime = ToolRuntime(
        [EchoTool(side_effect=SideEffectLevel.NETWORK)],
        max_side_effect=SideEffectLevel.LOCAL_STATE,
    )

    disabled = disabled_runtime.call(ToolCall("echo", {"message": "x"}))
    restricted = restricted_runtime.call(ToolCall("echo", {"message": "x"}))

    assert disabled.ok is False
    assert disabled.status is ToolExecutionStatus.DENIED
    assert disabled.error == "Tool is disabled: echo"
    assert restricted.ok is False
    assert "exceeds allowed level" in restricted.error
    assert restricted.status is ToolExecutionStatus.DENIED


def test_knowledge_search_tool_returns_hits(tmp_path: Path) -> None:
    source = make_vault(tmp_path)
    tool = KnowledgeSearchTool(source)

    result = tool.run({"query": "ăn rau xanh", "limit": 2})

    assert result.ok is True
    assert "Dinh dưỡng" in result.content
    assert result.data["hits"][0]["path"] == "wiki/nutrition.md"


def test_knowledge_search_tool_omits_line_range_when_source_has_none(
    tmp_path: Path,
) -> None:
    source = make_vault(tmp_path)
    result = KnowledgeSearchTool(source).run({"query": "ăn rau xanh", "limit": 2})

    hit = result.data["hits"][0]
    assert "line_start" not in hit
    assert "line_end" not in hit


def test_knowledge_read_tool_reads_relative_markdown_path(tmp_path: Path) -> None:
    source = make_vault(tmp_path)
    tool = KnowledgeReadTool(source)

    result = tool.run({"path": "wiki/nutrition.md"})

    assert result.ok is True
    assert "# Dinh dưỡng" in result.content
    assert result.data["path"] == "wiki/nutrition.md"
    assert result.data["tags"] == ["dinh-duong"]


def test_knowledge_read_tool_returns_bounded_line_receipt_and_continuation(
    tmp_path: Path,
) -> None:
    source = make_vault(tmp_path)
    note = tmp_path / "wiki" / "weekly.md"
    note.write_text(
        "\n".join(
            [
                "# Weekly review",
                "",
                "## Completed",
                "done one",
                "done two",
                "",
                "## Unfinished",
                "open one",
                "open two",
                "",
                "## Next",
                "next one",
            ]
        ),
        encoding="utf-8",
    )
    tool = KnowledgeReadTool(source, max_lines=3)

    first = tool.run(
        {
            "path": "wiki/weekly.md",
            "start_line": 7,
        }
    )
    second = tool.run(
        {
            "path": "wiki/weekly.md",
            "start_line": first.data["next_start_line"],
        }
    )

    assert "## Unfinished" in first.content
    assert "open two" in first.content
    assert "## Completed" not in first.content
    assert first.data["line_start"] == 7
    assert first.data["line_end"] == 9
    assert first.data["total_lines"] == 12
    assert first.data["next_start_line"] == 10
    assert first.data["complete"] is False
    assert first.data["truncated"] is True
    assert second.data["line_start"] == 10
    assert second.data["line_end"] == 12
    assert second.data["next_start_line"] is None
    assert second.data["complete"] is True


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({"path": "wiki/nutrition.md", "start_line": 0}, "invalid_start_line"),
        ({"path": "wiki/nutrition.md", "end_line": 0}, "invalid_end_line"),
        (
            {"path": "wiki/nutrition.md", "start_line": 2, "end_line": 1},
            "invalid_line_range",
        ),
        ({"path": "wiki/nutrition.md", "start_line": 99}, "start_line_out_of_range"),
    ],
)
def test_knowledge_read_tool_rejects_invalid_line_ranges(
    tmp_path: Path,
    arguments: dict[str, object],
    error: str,
) -> None:
    source = make_vault(tmp_path)

    result = ToolRuntime([KnowledgeReadTool(source)]).call(
        ToolCall("knowledge.read", arguments)
    )

    assert result.status is ToolExecutionStatus.INVALID
    assert result.error == error


def test_knowledge_read_tool_path_errors_are_returned_by_runtime(tmp_path: Path) -> None:
    source = make_vault(tmp_path)
    runtime = ToolRuntime([KnowledgeReadTool(source)])

    result = runtime.call(ToolCall("knowledge.read", {"path": "private/secret.md"}))

    assert result.status is ToolExecutionStatus.INVALID

    assert result.ok is False
    assert result.error == "invalid_path"


def test_knowledge_read_missing_path_uses_stable_error_code(tmp_path: Path) -> None:
    source = make_vault(tmp_path)
    result = KnowledgeReadTool(source).run({"path": "wiki/missing.md"})

    assert result.status is ToolExecutionStatus.NOT_FOUND
    assert result.error == "not_found"
