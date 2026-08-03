from __future__ import annotations

import json
from pathlib import Path

from eval import eval_grounding
from soca.knowledge import KnowledgeDocument, KnowledgeHit


class _RawOnlySource:
    def search(self, query: str, limit: int = 5) -> list[KnowledgeHit]:
        del limit
        if query != "Bayes":
            return []
        return [
            KnowledgeHit(
                KnowledgeDocument("answer", "wiki/answer.md", "Answer", "Bayes note"),
                score=1.0,
                snippet="Bayes note",
                retrieval_backend="lexical_custom",
                sparse_score=0.1,
            ),
            KnowledgeHit(
                KnowledgeDocument("distractor", "wiki/distractor.md", "Distractor", "Bayes distractor"),
                score=1.0,
                snippet="Bayes distractor",
                retrieval_backend="lexical_custom",
                sparse_score=1.0,
            )
        ]


def test_grounding_recall_uses_policy_accepted_paths(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "grounding.jsonl"
    rows = [
        {
            "id": "answerable",
            "family": "answerable",
            "split": "train",
            "query": "Bayes",
            "answerable": True,
            "relevant_paths": ["wiki/answer.md"],
        },
        {
            "id": "negative-validation",
            "family": "negative-validation",
            "split": "validation",
            "query": "missing",
            "answerable": False,
            "relevant_paths": [],
        },
        {
            "id": "negative-test",
            "family": "negative-test",
            "split": "test",
            "query": "missing",
            "answerable": False,
            "relevant_paths": [],
        },
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(eval_grounding, "_build_source", lambda **_: _RawOnlySource())

    report = eval_grounding.run_benchmark(
        vault=tmp_path,
        dataset=dataset,
        variant="cached_sparse",
        backend="fastembed",
    )

    assert "wiki/answer.md" in report["records"][0]["raw_paths"]
    assert "wiki/answer.md" not in report["records"][0]["accepted_paths"]
    assert report["answerable_retrieval_recall_at_5"]["successes"] == 1
    assert report["answerable_accepted_evidence_recall_at_5"]["successes"] == 0
