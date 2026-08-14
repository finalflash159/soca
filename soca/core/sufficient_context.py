"""Typed sufficient-context autorater for retrieval-grounded turns."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from soca.llm import LLMResult, StructuredLLMEngine
from soca.llm.providers import RemoteLLMError
from soca.tools import ToolResult


class SufficiencyStatus(StrEnum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


class SufficiencyPromptVariant(StrEnum):
    """Versioned prompt policies used by evaluation and controlled rollout."""

    STRICT_EXACT = "strict_exact"
    PAPER_DEFINITION = "paper_definition"
    BALANCED_EXAMPLES = "balanced_examples"


class SufficiencyAssessmentError(RuntimeError):
    """Raised when the autorater cannot return a trustworthy typed verdict."""

    def __init__(self, code: str) -> None:
        normalized = code.strip()
        if not normalized:
            raise ValueError("sufficiency error code must not be empty")
        super().__init__(normalized)
        self.code = normalized


@dataclass(frozen=True)
class RetrievedContext:
    evidence_id: str
    text: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.evidence_id.strip() or not self.text.strip():
            raise ValueError("retrieved context identity and text are required")
        object.__setattr__(self, "evidence_id", self.evidence_id.strip())
        object.__setattr__(self, "text", self.text.strip())
        object.__setattr__(self, "provenance", dict(self.provenance))


@dataclass(frozen=True)
class SufficiencyDecision:
    status: SufficiencyStatus
    confidence: float
    reason_code: str
    evidence_ids: tuple[str, ...] = ()
    model_id: str = ""
    prompt_sha256: str = ""
    provider_trace: Mapping[str, Any] = field(default_factory=dict)
    usage: Mapping[str, int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0 or not math.isfinite(self.confidence):
            raise ValueError("sufficiency confidence must be finite and between zero and one")
        if not _valid_reason_code(self.reason_code):
            raise ValueError("sufficiency reason code must be stable snake_case")
        if any(not item.strip() for item in self.evidence_ids):
            raise ValueError("sufficiency evidence ids must not be empty")
        object.__setattr__(self, "provider_trace", dict(self.provider_trace))
        object.__setattr__(self, "usage", dict(self.usage))

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "confidence": self.confidence,
            "reason_code": self.reason_code,
            "evidence_ids": list(self.evidence_ids),
            "model_id": self.model_id,
            "prompt_sha256": self.prompt_sha256,
            "provider_trace": dict(self.provider_trace),
            "usage": dict(self.usage),
        }


class ContextSufficiencyAssessor(Protocol):
    def assess(
        self,
        question: str,
        contexts: tuple[RetrievedContext, ...],
    ) -> SufficiencyDecision: ...


SUFFICIENCY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sufficient": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason_code": {
            "type": "string",
            "description": "Stable snake_case reason code; no free-form analysis.",
        },
    },
    "required": ["sufficient", "confidence", "reason_code"],
    "additionalProperties": False,
}


def _valid_reason_code(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip()
    return (
        bool(normalized)
        and len(normalized) <= 80
        and normalized[0].isalpha()
        and all(character.islower() or character.isdigit() or character == "_" for character in normalized)
    )


def parse_sufficiency_response(raw: str) -> SufficiencyDecision:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SufficiencyAssessmentError("invalid_output") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "sufficient",
        "confidence",
        "reason_code",
    }:
        raise SufficiencyAssessmentError("invalid_output")
    sufficient = payload["sufficient"]
    confidence = payload["confidence"]
    reason_code = payload["reason_code"]
    if (
        not isinstance(sufficient, bool)
        or isinstance(confidence, bool)
        or not isinstance(confidence, int | float)
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
        or not _valid_reason_code(reason_code)
    ):
        raise SufficiencyAssessmentError("invalid_output")
    return SufficiencyDecision(
        status=(
            SufficiencyStatus.SUFFICIENT
            if sufficient
            else SufficiencyStatus.INSUFFICIENT
        ),
        confidence=float(confidence),
        reason_code=str(reason_code).strip(),
    )


def retrieved_contexts_from_tool_results(
    results: tuple[ToolResult, ...],
) -> tuple[RetrievedContext, ...]:
    """Extract only selected knowledge evidence from typed tool receipts."""
    contexts: list[RetrievedContext] = []
    seen: set[tuple[str, str, int | None, int | None]] = set()
    for result in results:
        if not result.ok or result.name not in {"knowledge.search", "knowledge.read"}:
            continue
        if result.name == "knowledge.search":
            raw_hits = result.data.get("hits")
            if not isinstance(raw_hits, list):
                continue
            for raw_hit in raw_hits:
                if not isinstance(raw_hit, dict):
                    continue
                path = str(raw_hit.get("path", "")).strip()
                title = str(raw_hit.get("title", path)).strip()
                snippet = str(raw_hit.get("snippet", "")).strip()
                line_start = raw_hit.get("line_start")
                line_end = raw_hit.get("line_end")
                if not path or not snippet:
                    continue
                identity = (
                    path,
                    snippet,
                    line_start if isinstance(line_start, int) else None,
                    line_end if isinstance(line_end, int) else None,
                )
                if identity in seen:
                    continue
                seen.add(identity)
                provenance: dict[str, Any] = {"path": path, "title": title}
                if isinstance(line_start, int) and not isinstance(line_start, bool):
                    provenance["line_start"] = line_start
                if isinstance(line_end, int) and not isinstance(line_end, bool):
                    provenance["line_end"] = line_end
                contexts.append(
                    RetrievedContext(
                        evidence_id=f"K{len(contexts) + 1}",
                        text=snippet,
                        provenance=provenance,
                    )
                )
            continue

        path = str(result.data.get("path", "")).strip()
        content = result.content.strip()
        if not path or not content:
            continue
        line_start = result.data.get("line_start")
        line_end = result.data.get("line_end")
        identity = (
            path,
            content,
            line_start if isinstance(line_start, int) else None,
            line_end if isinstance(line_end, int) else None,
        )
        if identity in seen:
            continue
        seen.add(identity)
        provenance = {"path": path, "title": str(result.data.get("title", path))}
        if isinstance(line_start, int) and not isinstance(line_start, bool):
            provenance["line_start"] = line_start
        if isinstance(line_end, int) and not isinstance(line_end, bool):
            provenance["line_end"] = line_end
        contexts.append(
            RetrievedContext(
                evidence_id=f"K{len(contexts) + 1}",
                text=content,
                provenance=provenance,
            )
        )
    return tuple(contexts)


class SufficientContextAutorater:
    """Run one bounded structured model call over the selected evidence only."""

    def __init__(
        self,
        llm: StructuredLLMEngine,
        *,
        model_id: str,
        max_chars: int = 12_000,
        max_tokens: int = 96,
        prompt_variant: SufficiencyPromptVariant | str = SufficiencyPromptVariant.PAPER_DEFINITION,
    ) -> None:
        if not isinstance(llm, StructuredLLMEngine):
            raise TypeError("sufficient-context autorater requires structured output")
        if not model_id.strip():
            raise ValueError("sufficient-context model id is required")
        if max_chars < 512 or max_tokens < 16:
            raise ValueError("sufficient-context limits are too small")
        try:
            resolved_variant = SufficiencyPromptVariant(prompt_variant)
        except ValueError as exc:
            raise ValueError(f"unknown sufficient-context prompt variant: {prompt_variant}") from exc
        self.llm = llm
        self.model_id = model_id.strip()
        self.max_chars = max_chars
        self.max_tokens = max_tokens
        self.prompt_variant = resolved_variant

    def assess(
        self,
        question: str,
        contexts: tuple[RetrievedContext, ...],
    ) -> SufficiencyDecision:
        question = question.strip()
        if not question:
            raise SufficiencyAssessmentError("empty_question")
        if not contexts:
            return SufficiencyDecision(
                status=SufficiencyStatus.INSUFFICIENT,
                confidence=1.0,
                reason_code="no_context",
                model_id=self.model_id,
            )
        prompt = self._prompt(question, contexts)
        try:
            result = self.llm.generate_structured(
                prompt,
                schema_name="soca_sufficient_context",
                schema=SUFFICIENCY_SCHEMA,
                max_tokens=self.max_tokens,
                temperature=0.0,
                top_p=1.0,
                inject_persona=False,
                zero_data_retention=True,
            )
        except RemoteLLMError as exc:
            category = f"provider_{exc.category}"
            if not _valid_reason_code(category):
                category = "provider_unknown"
            raise SufficiencyAssessmentError(category) from exc
        except Exception as exc:  # noqa: BLE001 - provider boundary is fail-closed
            raise SufficiencyAssessmentError("provider_unavailable") from exc
        if not isinstance(result, LLMResult):
            raise SufficiencyAssessmentError("invalid_result_type")
        parsed = parse_sufficiency_response(result.text)
        return SufficiencyDecision(
            status=parsed.status,
            confidence=parsed.confidence,
            reason_code=parsed.reason_code,
            evidence_ids=tuple(context.evidence_id for context in contexts),
            model_id=self.model_id,
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            provider_trace=dict(result.provider_trace),
            usage={
                "prompt_tokens": result.n_prompt_tokens,
                "completion_tokens": result.n_completion_tokens,
                "ttft_ms": result.ttft_ms,
                "total_latency_ms": result.total_latency_ms,
            },
        )

    def _prompt(
        self,
        question: str,
        contexts: tuple[RetrievedContext, ...],
    ) -> str:
        prefix = self._prompt_prefix() + f"Question: {question}\n"
        remaining = self.max_chars - len(prefix)
        if remaining < 128:
            raise SufficiencyAssessmentError("context_budget_exceeded")
        sections: list[str] = []
        for context in contexts:
            header = f"\nEvidence {context.evidence_id}:\n"
            budget = remaining - len(header)
            if budget <= 0:
                break
            text = context.text[:budget]
            sections.append(header + text)
            remaining -= len(header) + len(text)
            if len(text) < len(context.text):
                break
        if not sections:
            raise SufficiencyAssessmentError("context_budget_exceeded")
        return prefix + "".join(sections) + "\nOutput:"

    def _prompt_prefix(self) -> str:
        output_contract = (
            "Chỉ trả về object JSON theo schema; không đưa phân tích hoặc nội dung "
            "bằng chứng vào output.\n\n"
        )
        if self.prompt_variant is SufficiencyPromptVariant.STRICT_EXACT:
            return (
                "Bạn là bộ kiểm tra sufficient-context độc lập theo nguyên tắc closed-world "
                "exact entailment. Chỉ đánh dấu sufficient khi bằng chứng trực tiếp hỗ trợ một "
                "câu trả lời ngắn cho TOÀN BỘ câu hỏi, gồm đúng chủ thể, đối tượng, vai trò, "
                "hành động, thời gian và mọi ràng buộc được hỏi. Đánh dấu insufficient nếu "
                "bất kỳ tiền đề, thực thể, quan hệ hoặc ràng buộc nào bị thiếu, sai, đảo vai, "
                "mâu thuẫn hoặc mơ hồ. Không sửa hộ câu hỏi, không thay một khái niệm bằng "
                "khái niệm gần nghĩa, không dùng kiến thức ngoài bằng chứng. Passage cùng chủ "
                "đề hoặc trả lời được một câu gần giống vẫn là insufficient. "
                + output_contract
                + "Ví dụ:\n"
                "Question: Ai là tác giả của tài liệu?\n"
                "Evidence K1: Tài liệu mô tả thuật toán nhưng không ghi tác giả.\n"
                'Output: {"sufficient":false,"confidence":0.98,'
                '"reason_code":"missing_requested_fact"}\n\n'
                "Question: An đã yêu cầu Bình làm gì sau khi Bình yêu cầu An rời đi?\n"
                "Evidence K1: An yêu cầu Bình rời đi.\n"
                'Output: {"sufficient":false,"confidence":0.99,'
                '"reason_code":"relation_role_mismatch"}\n\n'
            )

        definition = (
            "Bạn đánh giá context theo định nghĩa sufficient-context: một người đọc cẩn "
            "trọng có thể tạo ra câu trả lời đúng và dứt khoát cho toàn bộ câu hỏi chỉ từ "
            "bằng chứng đã cho. Cho phép kết hợp nhiều đoạn và suy luận hợp lý được bằng "
            "chứng hỗ trợ; không đòi câu trả lời phải xuất hiện nguyên văn. Đánh dấu "
            "insufficient khi thông tin cần thiết bị thiếu, mơ hồ, mâu thuẫn, sai thực thể, "
            "sai thời gian hoặc sai quan hệ. Không dùng kiến thức ngoài bằng chứng. "
            + output_contract
            + "Ví dụ đủ:\n"
            "Question: Lan sống ở đâu?\n"
            "Evidence K1: Lan chuyển nhà đến Huế và vẫn ở đó.\n"
            'Output: {"sufficient":true,"confidence":0.95,'
            '"reason_code":"supported_inference"}\n\n'
        )
        if self.prompt_variant is SufficiencyPromptVariant.PAPER_DEFINITION:
            return definition
        return (
            definition
            + "Ví dụ thiếu — cùng chủ đề nhưng sai quan hệ và đảo vai:\n"
            "Question: An yêu cầu Bình làm gì?\n"
            "Evidence K1: Bình yêu cầu An rời đi.\n"
            'Output: {"sufficient":false,"confidence":0.98,'
            '"reason_code":"relation_role_mismatch"}\n\n'
        )


__all__ = [
    "ContextSufficiencyAssessor",
    "RetrievedContext",
    "SUFFICIENCY_SCHEMA",
    "SufficiencyAssessmentError",
    "SufficiencyDecision",
    "SufficiencyPromptVariant",
    "SufficiencyStatus",
    "SufficientContextAutorater",
    "parse_sufficiency_response",
    "retrieved_contexts_from_tool_results",
]
