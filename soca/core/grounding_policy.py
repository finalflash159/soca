from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from soca.core.answer_validation import AnswerValidationDecision
from soca.core.evidence import EvidenceBundleDecision, EvidenceDecision

AnswerPolicyName = Literal[
    "free_chat",
    "grounded",
    "abstain",
    "conflict_disclosure",
    "retrieval_unavailable",
]
ValidationAction = Literal["allow", "repair", "block"]
GROUNDING_POLICY_VERSION = "grounding-v1"


@dataclass(frozen=True)
class GroundingTurnPolicy:
    name: AnswerPolicyName
    reason: str
    requires_citations: bool
    version: str = GROUNDING_POLICY_VERSION

    def validation_action(
        self,
        decision: AnswerValidationDecision,
        *,
        repair_attempted: bool,
    ) -> ValidationAction:
        if not self.requires_citations:
            return "allow"
        if decision.status == "valid":
            return "allow"
        if decision.status == "partial" and self.name != "conflict_disclosure":
            return "allow"
        if decision.status in {"missing", "partial", "invalid", "not_applicable"}:
            return "block" if repair_attempted else "repair"
        return "block"

    @property
    def block_message(self) -> str:
        return (
            "Mình chưa thể gửi câu trả lời này vì phần dẫn nguồn chưa hợp lệ. "
            "Bằng chứng gốc vẫn được giữ nguyên; bạn có thể thử hỏi lại cụ thể hơn."
        )


def select_grounding_policy(
    decisions: tuple[EvidenceDecision, ...],
    bundle: EvidenceBundleDecision | None,
) -> GroundingTurnPolicy:
    if not decisions:
        return GroundingTurnPolicy("free_chat", "no_retrieval_evidence", False)

    usable = tuple(
        decision
        for decision in decisions
        if decision.hit_count > 0 and decision.status in {"supported", "weak"}
    )
    if bundle is not None and bundle.status == "conflicting":
        return GroundingTurnPolicy(
            "conflict_disclosure",
            bundle.reason,
            bool(usable),
        )
    if (
        bundle is not None
        and bundle.status == "unknown"
        and len({decision.source for decision in usable}) > 1
    ):
        return GroundingTurnPolicy(
            "conflict_disclosure",
            "multiple_sources_not_reconciled",
            True,
        )
    if usable:
        return GroundingTurnPolicy("grounded", "usable_evidence", True)
    if all(decision.status == "unavailable" for decision in decisions):
        return GroundingTurnPolicy(
            "retrieval_unavailable",
            "all_requested_sources_unavailable",
            False,
        )
    return GroundingTurnPolicy("abstain", "no_usable_evidence", False)


def aggregate_evidence_status(decisions: tuple[EvidenceDecision, ...]) -> str:
    if not decisions:
        return "not_requested"
    statuses = {decision.status for decision in decisions}
    if "conflicting" in statuses:
        return "conflicting"
    if "supported" in statuses:
        return "supported"
    if "weak" in statuses:
        return "weak"
    if statuses == {"unavailable"}:
        return "unavailable"
    return "insufficient"


__all__ = [
    "AnswerPolicyName",
    "GROUNDING_POLICY_VERSION",
    "GroundingTurnPolicy",
    "ValidationAction",
    "aggregate_evidence_status",
    "select_grounding_policy",
]
