from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval import eval_retrieval_intent as evaluation


def _signal(case_id: str, split: str, score: float, relevant: bool) -> evaluation.IntentSignal:
    return evaluation.IntentSignal(split, relevant, True, False, score, 0.1)


def test_intent_metrics_choose_threshold_and_evaluate_validation() -> None:
    signals = (
        _signal("a", "train", 0.9, True),
        _signal("b", "train", 0.1, False),
        _signal("c", "validation", 0.95, True),
        _signal("d", "validation", 0.2, False),
    )
    report = evaluation.evaluate_signals(signals)
    assert report["threshold"] == 0.9
    assert report["validation"]["f1"] == 1.0


def test_load_cases_requires_train_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "intent.jsonl"
    path.write_text(
        json.dumps({"id": "a", "split": "train", "text": "hello", "expected": False}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="train and validation"):
        evaluation.load_cases(path)
