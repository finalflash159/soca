from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol

from soca.knowledge.base import KnowledgeHit

VoiceKnowledgeMode = Literal["off", "intent", "always"]
QUESTION_RE = re.compile(
    r"\b(là gì|la gi|như thế nào|nhu the nao|thế nào|the nao|ở đâu|o dau|"
    r"khi nào|khi nao|bao nhiêu|bao nhieu|tại sao|tai sao|vì sao|vi sao)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class IntentDecision:
    use_knowledge: bool
    reason: str
    score: float | None
    hits: tuple[KnowledgeHit, ...]


class RetrievalBatchLike(Protocol):
    hits: tuple[KnowledgeHit, ...]
    max_dense_score: float | None


class RetrievalSourceLike(Protocol):
    def retrieve(self, query: str, *, limit: int) -> RetrievalBatchLike: ...


class RetrievalIntentGate:
    def __init__(self, source: RetrievalSourceLike, *, threshold: float) -> None:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("intent threshold must be a number")
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("intent threshold must be between 0 and 1")
        self.source = source
        self.threshold = float(threshold)

    def evaluate(self, query: str, *, limit: int) -> IntentDecision:
        batch = self.source.retrieve(query, limit=limit)
        hits = tuple(batch.hits)
        if not hits:
            return IntentDecision(False, "no_hits", batch.max_dense_score, hits)
        if QUESTION_RE.search(query):
            return IntentDecision(True, "question_marker", batch.max_dense_score, hits)
        score = batch.max_dense_score
        if score is None:
            return IntentDecision(False, "no_dense_signal", None, hits)
        if score >= self.threshold:
            return IntentDecision(True, "dense_threshold", score, hits)
        return IntentDecision(False, "below_threshold", score, hits)
