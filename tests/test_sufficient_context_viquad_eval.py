from __future__ import annotations

from dataclasses import dataclass

import pytest

from eval.eval_sufficient_context_viquad import (
    DATASET_REVISION,
    DATASET_SPLIT,
    LabeledContext,
    aggregate_results,
    evaluate_contexts,
    load_labeled_contexts,
    select_balanced_contexts,
    wilson_interval,
)
from soca.core.sufficient_context import (
    RetrievedContext,
    SufficiencyAssessmentError,
    SufficiencyDecision,
    SufficiencyStatus,
)


def _rows() -> list[dict[str, object]]:
    return [
        {
            "id": "answerable-2",
            "context": "Lan sống ở Huế.",
            "question": "Lan sống ở đâu?",
            "is_impossible": False,
        },
        {
            "id": "impossible-2",
            "context": "Lan sống ở Huế.",
            "question": "Lan sinh năm nào?",
            "is_impossible": True,
        },
        {
            "id": "answerable-1",
            "context": "Minh làm việc ở Đà Nẵng.",
            "question": "Minh làm việc ở đâu?",
            "is_impossible": False,
        },
        {
            "id": "impossible-1",
            "context": "Minh làm việc ở Đà Nẵng.",
            "question": "Minh bao nhiêu tuổi?",
            "is_impossible": True,
        },
    ]


def test_loader_requires_pinned_validation_schema() -> None:
    contexts = load_labeled_contexts(_rows())

    assert DATASET_REVISION == "406f09a45cc106a8f7b7fd0c25078883fe58cb1f"
    assert DATASET_SPLIT == "validation"
    assert [item.expected_sufficient for item in contexts] == [True, False, True, False]

    with pytest.raises(ValueError, match="is_impossible"):
        load_labeled_contexts([{"id": "bad", "context": "x", "question": "y"}])


def test_balanced_selection_is_deterministic_and_stratified() -> None:
    contexts = load_labeled_contexts(_rows())

    first = select_balanced_contexts(contexts, per_class=2, seed=73)
    second = select_balanced_contexts(tuple(reversed(contexts)), per_class=2, seed=73)

    assert [item.case_id for item in first] == [item.case_id for item in second]
    assert sum(item.expected_sufficient for item in first) == 2
    assert len(first) == 4


def test_balanced_selection_rejects_underpowered_class() -> None:
    with pytest.raises(ValueError, match="expected_sufficient=False"):
        select_balanced_contexts(
            (LabeledContext("only", "q", "c", True),),
            per_class=1,
            seed=1,
        )


@dataclass
class _Assessor:
    verdicts: dict[str, bool]

    def assess(
        self,
        question: str,
        contexts: tuple[RetrievedContext, ...],
    ) -> SufficiencyDecision:
        del question
        verdict = self.verdicts[contexts[0].evidence_id]
        return SufficiencyDecision(
            status=(
                SufficiencyStatus.SUFFICIENT
                if verdict
                else SufficiencyStatus.INSUFFICIENT
            ),
            confidence=0.9,
            reason_code="fixture_verdict",
            evidence_ids=(contexts[0].evidence_id,),
            model_id="fixture-model",
            usage={"prompt_tokens": 10, "completion_tokens": 4, "total_latency_ms": 8.0},
        )


def test_evaluation_aggregates_confusion_usage_and_gate() -> None:
    contexts = load_labeled_contexts(_rows())
    assessor = _Assessor(
        {
            "answerable-1": True,
            "answerable-2": False,
            "impossible-1": False,
            "impossible-2": True,
        }
    )

    results = evaluate_contexts(contexts, assessor)
    aggregate = aggregate_results(
        results,
        false_sufficient_max=0.5,
        sufficient_recall_min=0.5,
    )

    assert aggregate["confusion_matrix"] == {"tp": 1, "fn": 1, "fp": 1, "tn": 1}
    assert aggregate["false_sufficient_rate"] == 0.5
    assert aggregate["sufficient_recall"] == 0.5
    assert aggregate["usage"] == {"completion_tokens": 16, "prompt_tokens": 40}
    assert aggregate["latency_ms"]["mean"] == 8.0
    assert aggregate["gate"]["passed"] is True
    assert all("question" not in result.as_public_dict() for result in results)
    assert all("context" not in result.as_public_dict() for result in results)


class _FailingAssessor:
    def assess(
        self,
        question: str,
        contexts: tuple[RetrievedContext, ...],
    ) -> SufficiencyDecision:
        del question, contexts
        raise SufficiencyAssessmentError("provider_unavailable")


def test_evaluator_records_typed_failure_and_fails_closed() -> None:
    result = evaluate_contexts(load_labeled_contexts(_rows()[:1]), _FailingAssessor())[0]
    aggregate = aggregate_results(
        (result,),
        false_sufficient_max=0.05,
        sufficient_recall_min=0.9,
    )

    assert result.error_code == "provider_unavailable"
    assert result.predicted_sufficient is None
    assert aggregate["gate"]["passed"] is False
    assert aggregate["gate"]["reasons"] == [
        "assessment_failures",
        "missing_insufficient_class",
        "missing_sufficient_class",
    ]


def test_wilson_interval_handles_zero_denominator() -> None:
    assert wilson_interval(0, 0) == (None, None)
    low, high = wilson_interval(0, 115)
    assert low == 0.0
    assert high is not None and high < 0.04
