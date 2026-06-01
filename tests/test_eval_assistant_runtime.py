from __future__ import annotations

from pathlib import Path

from eval.eval_assistant_runtime import (
    DEFAULT_PROMPT_PATH,
    evaluate_case,
    load_cases,
    run_eval,
)


def test_load_assistant_runtime_cases_reads_expected_route() -> None:
    cases = load_cases(DEFAULT_PROMPT_PATH, limit=1)

    assert cases[0].case_id == "time_001"
    assert cases[0].expected_route.value == "tool_direct"


def test_assistant_runtime_eval_default_suite_passes() -> None:
    report = run_eval(load_cases(DEFAULT_PROMPT_PATH))

    assert report["summary"]["pass_rate"] == 1.0
    assert report["summary"]["unsafe_to_llm_rate"] == 0.0
    assert report["summary"]["disabled_tool_execution_rate"] == 0.0


def test_disabled_tool_case_blocks_before_tool_execution() -> None:
    case = next(
        case for case in load_cases(DEFAULT_PROMPT_PATH) if case.case_id == "knowledge_disabled_001"
    )

    sample = evaluate_case(case)

    assert sample.passed is True
    assert sample.tool_calls == ("knowledge.search",)
    assert sample.tool_results == ()
    assert "tool_disabled" in sample.guardrail_reasons


def test_private_path_case_never_reaches_tool_or_llm() -> None:
    case = next(case for case in load_cases(DEFAULT_PROMPT_PATH) if case.case_id == "private_path_001")

    sample = evaluate_case(case)

    assert sample.passed is True
    assert sample.used_tool is False
    assert sample.used_llm is False
    assert sample.tool_calls == ()
    assert "blocked_path_prefix" in sample.guardrail_reasons


def test_load_cases_rejects_unknown_route(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"id":"bad","category":"x","input":"x","expected_route":"not_real","should_block":false}\n',
        encoding="utf-8",
    )

    try:
        load_cases(path)
    except ValueError as exc:
        assert "not_real" in str(exc)
    else:
        raise AssertionError("Expected invalid RuntimeRoute to raise ValueError")
