from __future__ import annotations

import json
from pathlib import Path

from eval.eval_memory_compaction import evaluate as evaluate_compaction
from eval.eval_memory_lifecycle import evaluate as evaluate_lifecycle
from eval.eval_router_ttfa import compare
from eval.eval_semantic_router import evaluate as evaluate_semantic
from eval.eval_tool_router import evaluate as evaluate_router


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_router_eval_scores_deterministic_cases(tmp_path: Path) -> None:
    path = tmp_path / "router.jsonl"
    _write_jsonl(
        path,
        [
            {"id": "one", "query": "knowledge: bayes", "expected_tool": "knowledge.search"},
            {"id": "two", "query": "hello", "expected_tool": "none"},
        ],
    )
    result = evaluate_router(path)
    assert result["accuracy"] == 1.0


def test_semantic_eval_requires_predictions_for_quality(tmp_path: Path) -> None:
    path = tmp_path / "semantic.jsonl"
    _write_jsonl(path, [{"id": "one", "expected_tool": "none"}])
    assert evaluate_semantic(path)["coverage"] == 0.0


def test_ttfa_comparison_and_compaction_are_bounded(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text('{"latency_p50_ms": 100}', encoding="utf-8")
    candidate.write_text('{"latency_p50_ms": 105}', encoding="utf-8")
    assert compare(baseline, candidate, 10.0)["within_budget"] is True
    result = evaluate_compaction(20, 4, 200)
    assert result["recent_turn_count"] == 4
    lifecycle = evaluate_lifecycle(tmp_path / "lifecycle")
    assert lifecycle["episode_round_trip"] is True
    assert lifecycle["proposal_approved"] is True
