from __future__ import annotations

from eval.eval_knowledge_memory import (
    DEFAULT_PROMPT_PATH,
    evaluate_case,
    load_cases,
    run_eval,
)


def test_knowledge_memory_eval_default_suite_passes() -> None:
    report = run_eval(load_cases(DEFAULT_PROMPT_PATH))

    assert report["summary"]["pass_rate"] == 1.0
    assert report["summary"]["context_budget_violation_rate"] == 0.0


def test_knowledge_context_case_reports_expected_hit_and_citation() -> None:
    case = next(case for case in load_cases(DEFAULT_PROMPT_PATH) if case.case_id == "km_knowledge_001")

    sample = evaluate_case(case)

    assert sample.passed is True
    assert sample.details["hit_paths"][0] == "wiki/dinh-duong/chat-dam.md"
    assert "wiki/dinh-duong/chat-dam.md" in sample.details["citation_paths"]


def test_private_path_guard_case_blocks_expected_prefix() -> None:
    case = next(case for case in load_cases(DEFAULT_PROMPT_PATH) if case.case_id == "km_path_private_001")

    sample = evaluate_case(case)

    assert sample.passed is True
    assert sample.details["action"] == "block"
    assert sample.details["reason"] == "blocked_path_prefix"


def test_memory_context_includes_profile_and_recent_session() -> None:
    case = next(case for case in load_cases(DEFAULT_PROMPT_PATH) if case.case_id == "km_memory_001")

    sample = evaluate_case(case)

    assert sample.passed is True
    assert sample.details["memory_chars"] > 0
    assert sample.details["session_chars"] > 0
