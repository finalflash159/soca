from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

from soca.tools.base import (
    SIDE_EFFECT_RANK,
    SideEffectLevel,
    ToolCall,
    ToolResult,
    ToolRuntime,
    validate_arguments,
)


class GuardrailStage(Enum):
    INPUT = "input"
    RETRIEVAL = "retrieval"
    TOOL_INPUT = "tool_input"
    TOOL_OUTPUT = "tool_output"
    OUTPUT = "output"


class GuardrailAction(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"


@dataclass(frozen=True)
class GuardrailEvent:
    stage: GuardrailStage
    action: GuardrailAction
    reason: str = ""
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return self.action == GuardrailAction.BLOCK


@dataclass(frozen=True)
class GuardrailPolicy:
    allowed_knowledge_prefixes: tuple[str, ...] = ("wiki/",)
    allowed_memory_paths: tuple[str, ...] = ("memory/core.json",)
    blocked_path_prefixes: tuple[str, ...] = ("raw/", "private/")
    blocked_path_parts: tuple[str, ...] = (".git", ".obsidian", ".trash")
    max_tool_side_effect: SideEffectLevel = SideEffectLevel.LOCAL_STATE
    require_citations_for_knowledge: bool = True


DEFAULT_POLICY = GuardrailPolicy()
MARKDOWN_PATH_RE = re.compile(r"(?P<path>[A-Za-z0-9_./\\-]+\.md)")

SYSTEM_PROMPT_PATTERNS = (
    "system prompt",
    "developer message",
    "hidden instruction",
    "chi dan he thong",
    "lenh he thong",
)
EXPLICIT_READ_PREFIXES = ("read:", "read ", "doc:", "doc ", "đọc:", "đọc ")
INJECTION_PATTERNS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "bo qua chi dan truoc",
    "bo qua moi chi dan",
    "hay bo qua chi dan",
    "system:",
    "developer:",
    "assistant:",
    "goi tool",
    "xoa file",
)
SUCCESS_CLAIM_PATTERNS = (
    "da dat",
    "da thuc hien",
    "xong roi",
    "minh da",
    "done",
)
REALTIME_CLAIM_PATTERNS = (
    "hien tai troi",
    "bay gio troi",
    "nhiet do hien tai",
    "thoi tiet hien tai",
)


def allow(stage: GuardrailStage, metadata: dict[str, Any] | None = None) -> GuardrailEvent:
    return GuardrailEvent(stage=stage, action=GuardrailAction.ALLOW, metadata=metadata or {})


def block(
    stage: GuardrailStage,
    reason: str,
    message: str = "",
    metadata: dict[str, Any] | None = None,
) -> GuardrailEvent:
    return GuardrailEvent(
        stage=stage,
        action=GuardrailAction.BLOCK,
        reason=reason,
        message=message,
        metadata=metadata or {},
    )


def warn(
    stage: GuardrailStage,
    reason: str,
    message: str = "",
    metadata: dict[str, Any] | None = None,
) -> GuardrailEvent:
    return GuardrailEvent(
        stage=stage,
        action=GuardrailAction.WARN,
        reason=reason,
        message=message,
        metadata=metadata or {},
    )


def normalize_vi(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    folded = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return folded.lower()


def contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    normalized = normalize_vi(text)
    return any(pattern in normalized for pattern in patterns)


def extract_markdown_paths(text: str) -> tuple[str, ...]:
    return tuple(match.group("path") for match in MARKDOWN_PATH_RE.finditer(text))


def _normalize_path(path: str) -> str:
    normalized = path.strip().strip("'\"`").replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def check_knowledge_read_path(
    path: str,
    policy: GuardrailPolicy = DEFAULT_POLICY,
) -> GuardrailEvent:
    normalized = _normalize_path(path)
    metadata = {"path": path, "normalized_path": normalized}

    if not normalized:
        return block(GuardrailStage.RETRIEVAL, "empty_path", metadata=metadata)
    if normalized.startswith("/"):
        return block(GuardrailStage.RETRIEVAL, "absolute_path", metadata=metadata)
    if normalized in policy.allowed_memory_paths:
        return block(
            GuardrailStage.RETRIEVAL,
            "outside_runtime_knowledge_scope",
            "Memory is read through the memory module, not knowledge.read.",
            metadata=metadata,
        )
    if not normalized.endswith(".md"):
        return block(GuardrailStage.RETRIEVAL, "non_markdown_path", metadata=metadata)

    parts = PurePosixPath(normalized).parts
    if ".." in parts:
        return block(GuardrailStage.RETRIEVAL, "path_traversal", metadata=metadata)
    if any(part.startswith(".") or part in policy.blocked_path_parts for part in parts):
        return block(GuardrailStage.RETRIEVAL, "blocked_path_part", metadata=metadata)
    if normalized.startswith(policy.blocked_path_prefixes):
        return block(GuardrailStage.RETRIEVAL, "blocked_path_prefix", metadata=metadata)
    if not normalized.startswith(policy.allowed_knowledge_prefixes):
        return block(
            GuardrailStage.RETRIEVAL,
            "outside_runtime_knowledge_scope",
            metadata=metadata,
        )

    return allow(GuardrailStage.RETRIEVAL, metadata=metadata)


def check_input_text(
    text: str,
    policy: GuardrailPolicy = DEFAULT_POLICY,
) -> GuardrailEvent:
    if not text.strip():
        return block(GuardrailStage.INPUT, "empty_input", "Mình chưa nghe rõ.")

    normalized = normalize_vi(text.strip())
    if any(normalized.startswith(normalize_vi(prefix)) for prefix in EXPLICIT_READ_PREFIXES):
        for path in extract_markdown_paths(text):
            path_event = check_knowledge_read_path(path, policy)
            if path_event.blocked:
                return GuardrailEvent(
                    stage=GuardrailStage.INPUT,
                    action=GuardrailAction.BLOCK,
                    reason=path_event.reason,
                    message=path_event.message,
                    metadata=path_event.metadata,
                )

    if contains_any(text, SYSTEM_PROMPT_PATTERNS):
        return block(
            GuardrailStage.INPUT,
            "system_prompt_extraction",
            "Mình không thể tiết lộ system prompt hoặc chỉ dẫn nội bộ.",
        )

    return allow(GuardrailStage.INPUT)


def check_untrusted_text(
    text: str,
    stage: GuardrailStage = GuardrailStage.RETRIEVAL,
) -> GuardrailEvent:
    normalized = normalize_vi(text)
    for pattern in INJECTION_PATTERNS:
        if pattern in normalized:
            return warn(
                stage,
                "prompt_injection_like_text",
                metadata={"pattern": pattern},
            )
    return allow(stage)


def check_tool_call(
    runtime: ToolRuntime,
    call: ToolCall,
    policy: GuardrailPolicy = DEFAULT_POLICY,
    *,
    user_text: str = "",
    knowledge_read_paths: Collection[str] = (),
    require_read_provenance: bool = False,
) -> GuardrailEvent:
    tool = runtime.get(call.name)
    metadata = {"tool": call.name}
    if tool is None:
        return block(GuardrailStage.TOOL_INPUT, "unknown_tool", metadata=metadata)

    spec = tool.spec
    metadata["side_effect"] = spec.side_effect.value
    if not spec.enabled:
        return block(GuardrailStage.TOOL_INPUT, "tool_disabled", metadata=metadata)

    if SIDE_EFFECT_RANK[spec.side_effect] > SIDE_EFFECT_RANK[policy.max_tool_side_effect]:
        return block(GuardrailStage.TOOL_INPUT, "side_effect_not_allowed", metadata=metadata)

    validation_error = validate_arguments(spec.input_schema, call.arguments)
    if validation_error:
        return block(
            GuardrailStage.TOOL_INPUT,
            "invalid_tool_arguments",
            validation_error,
            metadata=metadata,
        )

    if call.name == "knowledge.read":
        path = str(call.arguments.get("path", ""))
        path_event = check_knowledge_read_path(path, policy)
        if path_event.blocked:
            return GuardrailEvent(
                stage=GuardrailStage.TOOL_INPUT,
                action=GuardrailAction.BLOCK,
                reason=path_event.reason,
                message=path_event.message,
                metadata={**metadata, **path_event.metadata},
            )
        if require_read_provenance:
            normalized_path = path_event.metadata.get("normalized_path")
            explicit_paths = {
                explicit_event.metadata.get("normalized_path")
                for explicit_path in extract_markdown_paths(user_text)
                for explicit_event in (check_knowledge_read_path(explicit_path, policy),)
                if not explicit_event.blocked
            }
            known_paths = {
                _normalize_path(known_path)
                for known_path in knowledge_read_paths
                if isinstance(known_path, str) and known_path.strip()
            }
            if normalized_path not in explicit_paths | known_paths:
                return block(
                    GuardrailStage.TOOL_INPUT,
                    "knowledge_read_requires_provenance",
                    "Chưa có đường dẫn này từ câu hỏi hoặc receipt truy xuất trước đó.",
                    metadata={
                        **metadata,
                        **path_event.metadata,
                        "provenance": "missing_search_or_explicit_path",
                    },
                )

    return allow(GuardrailStage.TOOL_INPUT, metadata=metadata)


def knowledge_paths_from_results(results: Iterable[ToolResult]) -> tuple[str, ...]:
    """Collect paths established by typed navigation/search receipts.

    A catalog or search receipt can establish that a path exists. It does not
    make arbitrary vault paths readable; the caller still applies the normal
    path guard before executing ``knowledge.read``.
    """
    paths: set[str] = set()
    for result in results:
        if result.name == "knowledge.search":
            raw_paths = (
                item.get("path")
                for item in result.data.get("hits", [])
                if isinstance(item, dict)
            )
        elif result.name == "knowledge.inspect":
            raw_paths = (
                item.get("path")
                for item in result.data.get("documents", [])
                if isinstance(item, dict)
            )
        elif result.name == "knowledge.read":
            raw_paths = (result.data.get("path"),)
        else:
            continue
        for raw_path in raw_paths:
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            event = check_knowledge_read_path(raw_path)
            if not event.blocked:
                normalized = event.metadata.get("normalized_path")
                if isinstance(normalized, str) and normalized:
                    paths.add(normalized)
    return tuple(sorted(paths))


def check_tool_result(result: ToolResult) -> GuardrailEvent:
    metadata = {"tool": result.name}
    if not result.ok:
        return block(GuardrailStage.TOOL_OUTPUT, "tool_failed", result.error, metadata=metadata)

    untrusted_event = check_untrusted_text(result.content, stage=GuardrailStage.TOOL_OUTPUT)
    if untrusted_event.action == GuardrailAction.WARN:
        return untrusted_event

    return allow(GuardrailStage.TOOL_OUTPUT, metadata=metadata)


def check_final_output(
    text: str,
    *,
    knowledge_used: bool = False,
    citations: tuple[Any, ...] = (),
    tool_results: tuple[ToolResult, ...] = (),
    realtime_tool_used: bool = False,
    policy: GuardrailPolicy = DEFAULT_POLICY,
) -> GuardrailEvent:
    if policy.require_citations_for_knowledge and knowledge_used and not citations:
        return block(
            GuardrailStage.OUTPUT,
            "missing_citation",
            "Câu trả lời dùng knowledge nhưng thiếu nguồn trích dẫn.",
        )

    if any(not result.ok for result in tool_results) and contains_any(text, SUCCESS_CLAIM_PATTERNS):
        return block(
            GuardrailStage.OUTPUT,
            "false_tool_success_claim",
            "Mình chưa thực hiện được thao tác đó.",
        )

    if not realtime_tool_used and contains_any(text, REALTIME_CLAIM_PATTERNS):
        return block(
            GuardrailStage.OUTPUT,
            "unsupported_realtime_claim",
            "Mình chưa có dữ liệu thời gian thực cho câu này.",
        )

    return allow(GuardrailStage.OUTPUT)
