from __future__ import annotations

from typing import Literal

from soca.core.answer_validation import AnswerValidationDecision
from soca.core.evidence import EvidenceBundleDecision, EvidenceDecision, EvidenceStatus
from soca.core.grounding_policy import (
    GROUNDING_POLICY_VERSION,
    aggregate_evidence_status,
    select_grounding_policy,
)


def _decision(
    source: Literal["knowledge", "memory"],
    status: EvidenceStatus,
    *,
    hits: int = 1,
) -> EvidenceDecision:
    return EvidenceDecision(
        source=source,
        status=status,
        hit_count=hits,
        top_score=0.8 if hits else None,
        margin=None,
        rejected_count=0,
        reason="test",
    )


def test_multiple_unreconciled_sources_require_conflict_disclosure() -> None:
    decisions = (
        _decision("knowledge", "supported"),
        _decision("memory", "supported"),
    )
    bundle = EvidenceBundleDecision(
        "unknown",
        decisions,
        "multiple_supported_sources_unreconciled",
    )

    policy = select_grounding_policy(decisions, bundle)

    assert policy.name == "conflict_disclosure"
    assert policy.requires_citations is True
    assert policy.version == GROUNDING_POLICY_VERSION


def test_grounded_policy_repairs_once_then_blocks_invalid_provenance() -> None:
    decision = _decision("knowledge", "supported")
    policy = select_grounding_policy((decision,), None)
    validation = AnswerValidationDecision("missing", ("[K1]",), (), "no_provenance_label")

    assert policy.validation_action(validation, repair_attempted=False) == "repair"
    assert policy.validation_action(validation, repair_attempted=True) == "block"


def test_grounded_policy_allows_valid_subset_but_conflict_policy_requires_both_sources() -> None:
    decisions = (
        _decision("knowledge", "supported"),
        _decision("memory", "supported"),
    )
    partial = AnswerValidationDecision(
        "partial",
        ("[K1]", "[M1]"),
        ("[K1]",),
        "partial_provenance_labels",
    )

    grounded = select_grounding_policy((decisions[0],), None)
    conflict = select_grounding_policy(
        decisions,
        EvidenceBundleDecision("unknown", decisions, "unreconciled"),
    )

    assert grounded.validation_action(partial, repair_attempted=False) == "allow"
    assert conflict.validation_action(partial, repair_attempted=False) == "repair"


def test_empty_and_unavailable_evidence_choose_distinct_answer_policies() -> None:
    empty = _decision("knowledge", "insufficient", hits=0)
    unavailable = _decision("memory", "unavailable", hits=0)

    assert select_grounding_policy((empty,), None).name == "abstain"
    assert select_grounding_policy((unavailable,), None).name == "retrieval_unavailable"
    assert aggregate_evidence_status((empty,)) == "insufficient"
    assert aggregate_evidence_status((unavailable,)) == "unavailable"
