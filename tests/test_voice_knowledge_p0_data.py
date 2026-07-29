from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.eval_full_cascade import evaluate as evaluate_cascade
from eval.eval_full_cascade import load_dataset as load_route_dataset
from eval.eval_grounding import load_dataset as load_grounding_dataset
from eval.eval_memory_context_policy import load_cases as load_memory_policy_cases
from eval.eval_source_policy import evaluate as evaluate_sources
from eval.eval_source_policy import load_dataset as load_source_dataset

ROOT = Path(__file__).resolve().parents[1]
P0 = ROOT / "eval" / "prompts" / "p0"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_p0_route_dataset_is_family_split_and_not_demo_content() -> None:
    rows = load_route_dataset(P0 / "turn_routing_vi.jsonl")
    assert len(rows) == 66
    assert {row["split"] for row in rows} == {"train", "validation", "test"}
    assert all("knowledge_vault" not in row["query"] for row in rows)
    assert all("dinh-duong" not in row["query"] for row in rows)


def test_p0_source_and_grounding_contracts_are_frozen() -> None:
    source_rows = load_source_dataset(P0 / "retrieval_source_vi.jsonl")
    grounding_rows = load_grounding_dataset(P0 / "grounding_vi.jsonl")
    assert len(source_rows) == 40
    assert len(grounding_rows) == 20
    assert {row["expected_source_profile"] for row in source_rows} == {
        "knowledge",
        "memory",
        "both",
        "neither",
    }
    assert sum(row["answerable"] for row in grounding_rows) == 12
    assert all(
        row["relevant_paths"] == [] or all(path.startswith("wiki/") for path in row["relevant_paths"])
        for row in grounding_rows
    )


def test_p0_policy_and_parity_files_have_required_rows() -> None:
    policy_rows = [json.loads(line) for line in (P0 / "memory_context_policy_vi.jsonl").read_text().splitlines()]
    parity_rows = [json.loads(line) for line in (P0 / "voice_parity_vi.jsonl").read_text().splitlines()]
    assert len(policy_rows) == 16
    assert len(parity_rows) == 14
    assert {row["split"] for row in policy_rows} == {"train", "validation", "test"}
    assert {row["split"] for row in parity_rows} == {"train", "validation", "test"}
    assert {row["parity_id"] for row in parity_rows} == {f"vp-{index:03d}" for index in range(1, 8)}
    assert len(load_memory_policy_cases(P0 / "memory_context_policy_vi.jsonl")) == 16


def test_cascade_evaluator_scores_disposition_source_and_family(tmp_path: Path) -> None:
    dataset = tmp_path / "route.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        {"id": "train", "family": "train-family", "split": "train", "query": "x", "disposition": "smalltalk", "sources": []},
        {"id": "validation", "family": "validation-family", "split": "validation", "query": "y", "disposition": "direct_tool", "sources": [], "tool": "local_time.now"},
        {"id": "test", "family": "test-family", "split": "test", "query": "z", "disposition": "out_of_scope", "sources": []},
    ]
    _write_jsonl(dataset, rows)
    _write_jsonl(
        predictions,
        [
            {"id": "train", "disposition": "smalltalk", "sources": [], "tool": "none"},
            {"id": "validation", "disposition": "direct_tool", "sources": [], "tool": "local_time.now"},
            {"id": "test", "disposition": "out_of_scope", "sources": [], "tool": "none"},
        ],
    )
    result = evaluate_cascade(dataset, predictions)
    assert result["disposition_accuracy"] == 1.0
    assert result["source_exact_accuracy"] == 1.0
    assert result["tool_metrics"]["direct_tool_exact"]["rate"] == 1.0
    assert result["tool_metrics"]["unsupported_to_real_tool"]["rate"] == 0.0
    assert result["by_family"]["test-family"]["disposition_accuracy"] == 1.0

    _write_jsonl(predictions, rows[:2])
    with pytest.raises(ValueError, match="exactly match"):
        evaluate_cascade(dataset, predictions)

    _write_jsonl(predictions, rows + [{"id": "stale", "disposition": "smalltalk", "sources": []}])
    with pytest.raises(ValueError, match="exactly match"):
        evaluate_cascade(dataset, predictions)


def test_source_evaluator_reports_neither_without_collapsing_it(tmp_path: Path) -> None:
    dataset = tmp_path / "source.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    rows = [
        {"id": "train", "family": "train-family", "split": "train", "query": "x", "disposition": "retrieval_request", "sources": [], "expected_source_profile": "neither"},
        {"id": "validation", "family": "validation-family", "split": "validation", "query": "y", "disposition": "retrieval_request", "sources": ["knowledge"], "expected_source_profile": "knowledge"},
        {"id": "test", "family": "test-family", "split": "test", "query": "z", "disposition": "retrieval_request", "sources": ["knowledge", "memory"], "expected_source_profile": "both"},
    ]
    _write_jsonl(dataset, rows)
    _write_jsonl(
        predictions,
        [
            {"id": "train", "sources": []},
            {"id": "validation", "sources": ["knowledge"]},
            {"id": "test", "sources": ["knowledge", "memory"]},
        ],
    )
    result = evaluate_sources(dataset, predictions)
    assert result["source_exact_accuracy"]["rate"] == 1.0
    assert result["by_profile"]["neither"]["count"] == 1
    assert result["by_profile"]["neither"]["correct"] == 1
    assert result["by_profile"]["neither"]["precision"] == 1.0

    _write_jsonl(predictions, [
        {"id": "train", "sources": []},
        {"id": "validation", "sources": ["knowledge"]},
    ])
    with pytest.raises(ValueError, match="exactly match"):
        evaluate_sources(dataset, predictions)
