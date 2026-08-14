from __future__ import annotations

from dataclasses import dataclass

import pytest

from eval.eval_sufficient_context_viquad import (
    DATASET_REVISION,
    DATASET_SPLIT,
    DEFAULT_FALSE_SUFFICIENT_MAX,
    DEFAULT_PER_CLASS,
    DEFAULT_SUFFICIENT_RECALL_MIN,
    RELEASE_MINIMUM_REVIEWED_PER_CLASS,
    LabeledContext,
    aggregate_results,
    apply_reviewed_labels,
    build_parser,
    demote_non_release_run,
    evaluate_contexts,
    load_labeled_contexts,
    reasoning_options_for_provider,
    release_overrides,
    select_balanced_contexts,
    wilson_interval,
)
from soca.config import LlmSettings
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
        release_labels_reviewed=True,
        minimum_class_count=1,
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
        release_labels_reviewed=True,
        minimum_class_count=1,
    )

    assert result.error_code == "provider_unavailable"
    assert result.predicted_sufficient is None
    assert aggregate["gate"]["passed"] is False
    # A run where every call failed has nothing assessed, so both classes are
    # simultaneously empty and underpowered. Reporting all five reasons keeps the
    # artifact honest about *why* it cannot be read as evidence.
    assert aggregate["gate"]["reasons"] == [
        "underpowered_sufficient_class",
        "underpowered_insufficient_class",
        "assessment_failures",
        "missing_insufficient_class",
        "missing_sufficient_class",
    ]


def test_wilson_interval_handles_zero_denominator() -> None:
    assert wilson_interval(0, 0) == (None, None)
    low, high = wilson_interval(0, 115)
    assert low == 0.0
    assert high is not None and high < 0.04


def test_reviewed_labels_require_two_reviewer_consensus_and_full_coverage() -> None:
    contexts = load_labeled_contexts(_rows())
    reviewed = apply_reviewed_labels(
        contexts,
        {
            "dataset_revision": DATASET_REVISION,
            "label_definition": "sufficient_context_semantic_v1",
            "reviewers": ["reviewer-a", "reviewer-b"],
            "cases": [
                {
                    "case_id": item.case_id,
                    "expected_sufficient": not item.expected_sufficient,
                    "reviewer_labels": {
                        "reviewer-a": not item.expected_sufficient,
                        "reviewer-b": not item.expected_sufficient,
                    },
                }
                for item in contexts
            ],
        },
    )

    assert [item.expected_sufficient for item in reviewed] == [False, True, False, True]

    with pytest.raises(ValueError, match="at least two reviewers"):
        apply_reviewed_labels(
            contexts,
            {
                "dataset_revision": DATASET_REVISION,
                "label_definition": "sufficient_context_semantic_v1",
                "reviewers": ["reviewer-a"],
                "cases": [],
            },
        )

    with pytest.raises(ValueError, match="reviewer labels"):
        apply_reviewed_labels(
            contexts,
            {
                "dataset_revision": DATASET_REVISION,
                "label_definition": "sufficient_context_semantic_v1",
                "reviewers": ["reviewer-a", "reviewer-b"],
                "cases": [
                    {
                        "case_id": item.case_id,
                        "expected_sufficient": item.expected_sufficient,
                        "reviewer_labels": {"reviewer-a": item.expected_sufficient},
                    }
                    for item in contexts
                ],
            },
        )


def test_proxy_labels_cannot_pass_release_gate() -> None:
    contexts = load_labeled_contexts(_rows())
    results = evaluate_contexts(
        contexts,
        _Assessor({item.case_id: item.expected_sufficient for item in contexts}),
    )

    report = aggregate_results(
        results,
        false_sufficient_max=0.05,
        sufficient_recall_min=0.9,
        release_labels_reviewed=False,
        minimum_class_count=1,
    )

    assert report["gate"]["passed"] is False
    assert "proxy_labels_not_release_evidence" in report["gate"]["reasons"]


def test_cli_accepts_explicit_provider_prompt_variant_and_reviewed_labels() -> None:
    args = build_parser().parse_args(
        [
            "--provider",
            "gemini",
            "--model",
            "gemini-2.5-pro",
            "--prompt-variant",
            "paper_definition",
            "--reviewed-labels",
            "reviewed.json",
            "--autorater-max-tokens",
            "512",
        ]
    )

    assert args.provider == "gemini"
    assert args.prompt_variant == "paper_definition"
    assert args.reviewed_labels.name == "reviewed.json"
    assert args.autorater_max_tokens == 512


def test_provider_override_does_not_leak_openrouter_reasoning_transport() -> None:
    settings = LlmSettings(
        backend="remote",
        provider_key="openrouter",
        model_id="openai/test",
        max_tokens=2_048,
        reasoning_enabled=False,
        model_reasoning_supported=True,
        model_reasoning_parameter="reasoning",
    )

    assert reasoning_options_for_provider(
        settings,
        "openrouter",
        "openai/test",
    ) == (False, "reasoning")
    assert reasoning_options_for_provider(
        settings,
        "openrouter",
        "anthropic/other",
    ) == (None, None)
    # A different provider must not inherit the persisted capabilities either, or a
    # bake-off across providers would silently carry one provider's reasoning flags.
    assert reasoning_options_for_provider(
        settings,
        "gemini",
        "gemini-3.6-flash",
    ) == (None, None)


def test_release_gate_rejects_underpowered_reviewed_semantic_class() -> None:
    contexts = tuple(
        LabeledContext(item.case_id, item.question, item.context, True)
        for item in load_labeled_contexts(_rows())
    )
    results = evaluate_contexts(
        contexts,
        _Assessor({item.case_id: True for item in contexts}),
    )

    report = aggregate_results(
        results,
        false_sufficient_max=1.0,
        sufficient_recall_min=0.0,
        release_labels_reviewed=True,
        minimum_class_count=2,
    )

    assert report["gate"]["passed"] is False
    assert "underpowered_insufficient_class" in report["gate"]["reasons"]


def test_pinned_release_configuration_reports_no_overrides() -> None:
    assert release_overrides(
        false_sufficient_max=DEFAULT_FALSE_SUFFICIENT_MAX,
        sufficient_recall_min=DEFAULT_SUFFICIENT_RECALL_MIN,
        per_class=DEFAULT_PER_CLASS,
        minimum_reviewed_per_class=RELEASE_MINIMUM_REVIEWED_PER_CLASS,
        reviewed_labels_present=True,
    ) == ()


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("false_sufficient_max", 0.5, "relaxed_false_sufficient_max"),
        ("sufficient_recall_min", 0.5, "relaxed_sufficient_recall_min"),
        ("per_class", 5, "reduced_per_class"),
        ("minimum_reviewed_per_class", 1, "reduced_minimum_reviewed_per_class"),
        ("reviewed_labels_present", False, "proxy_labels_not_release_evidence"),
    ],
)
def test_every_loosened_knob_is_named_as_an_override(
    field: str,
    value: object,
    expected: str,
) -> None:
    # Each knob is individually sufficient to disqualify a run from being read as
    # release evidence; a caller must not be able to reach gate.passed by turning
    # one of them down.
    kwargs: dict[str, object] = {
        "false_sufficient_max": DEFAULT_FALSE_SUFFICIENT_MAX,
        "sufficient_recall_min": DEFAULT_SUFFICIENT_RECALL_MIN,
        "per_class": DEFAULT_PER_CLASS,
        "minimum_reviewed_per_class": RELEASE_MINIMUM_REVIEWED_PER_CLASS,
        "reviewed_labels_present": True,
    }
    kwargs[field] = value

    assert release_overrides(**kwargs) == (expected,)  # type: ignore[arg-type]


def test_tightening_a_threshold_is_not_an_override() -> None:
    # Stricter than release is still release-eligible: the gate only refuses to be
    # read as evidence when it was made *easier* to pass.
    assert release_overrides(
        false_sufficient_max=0.01,
        sufficient_recall_min=0.99,
        per_class=DEFAULT_PER_CLASS + 50,
        minimum_reviewed_per_class=RELEASE_MINIMUM_REVIEWED_PER_CLASS + 5,
        reviewed_labels_present=True,
    ) == ()


def test_dirty_or_overridden_runs_cannot_report_a_passing_gate() -> None:
    passing = {"gate": {"passed": True, "reasons": []}}

    demoted = demote_non_release_run(passing, overrides=("reduced_per_class",), dirty=False)
    assert demoted["gate"]["passed"] is False
    assert demoted["gate"]["reasons"] == ["reduced_per_class"]
    assert demoted["run_class"] == "diagnostic"

    dirty = demote_non_release_run({"gate": {"passed": True, "reasons": []}}, overrides=(), dirty=True)
    assert dirty["gate"]["passed"] is False
    assert dirty["gate"]["reasons"] == ["evaluation_source_dirty"]
    assert dirty["run_class"] == "diagnostic"


def test_clean_pinned_run_stays_a_release_run() -> None:
    report = demote_non_release_run(
        {"gate": {"passed": True, "reasons": []}},
        overrides=(),
        dirty=False,
    )
    assert report["gate"]["passed"] is True
    assert report["run_class"] == "release"
