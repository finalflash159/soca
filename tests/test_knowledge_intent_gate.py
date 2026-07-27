from __future__ import annotations

from types import SimpleNamespace

import pytest

from soca.knowledge.base import KnowledgeDocument, KnowledgeHit
from soca.knowledge.intent_gate import RetrievalIntentGate


def _hit() -> KnowledgeHit:
    return KnowledgeHit(
        KnowledgeDocument("a", "wiki/a.md", "A", "protein facts"), 0.5, "protein facts"
    )


@pytest.mark.parametrize("threshold", [-0.1, 1.1, True, "0.5"])
def test_gate_rejects_invalid_threshold(threshold: object) -> None:
    with pytest.raises(ValueError):
        RetrievalIntentGate(object(), threshold=threshold)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("score", "expected", "reason"),
    [
        (0.2, False, "below_threshold"),
        (0.5, True, "dense_threshold"),
        (0.8, True, "dense_threshold"),
    ],
)
def test_gate_uses_dense_threshold(score: float, expected: bool, reason: str) -> None:
    calls: list[str] = []

    class Source:
        def retrieve(self, query: str, *, limit: int):
            calls.append(query)
            return SimpleNamespace(hits=(_hit(),), max_dense_score=score)

    decision = RetrievalIntentGate(Source(), threshold=0.5).evaluate("protein", limit=3)
    assert decision.use_knowledge is expected
    assert decision.reason == reason
    assert calls == ["protein"]


def test_gate_does_not_override_low_dense_score_with_question_keywords() -> None:
    class Source:
        def retrieve(self, query: str, *, limit: int):
            return SimpleNamespace(hits=(_hit(),), max_dense_score=0.01)

    decision = RetrievalIntentGate(Source(), threshold=0.9).evaluate("Protein là gì?", limit=3)
    assert decision.use_knowledge is False
    assert decision.reason == "below_threshold"


def test_gate_no_hits_and_no_dense_signal_are_safe() -> None:
    class Source:
        def __init__(self, hits):
            self.hits = hits

        def retrieve(self, query: str, *, limit: int):
            return SimpleNamespace(hits=self.hits, max_dense_score=None)

    assert (
        RetrievalIntentGate(Source(()), threshold=0.5).evaluate("hello", limit=3).reason
        == "no_hits"
    )
    decision = RetrievalIntentGate(Source((_hit(),)), threshold=0.5).evaluate("hello", limit=3)
    assert decision.use_knowledge is False
    assert decision.reason == "no_dense_signal"
