from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from eval import eval_hybrid_retrieval as evaluation
from soca.knowledge.base import KnowledgeDocument, KnowledgeHit
from soca.knowledge.hybrid_source import DenseUnavailableError


def _case(case_id: str = "one", slice_name: str = "study") -> evaluation.RetrievalCase:
    return evaluation.RetrievalCase(
        case_id=case_id,
        slice_name=slice_name,
        query="bayes",
        relevant_paths=("wiki/bayes.md",),
    )


def test_metrics_deduplicate_retrieved_paths_and_match_ir_definitions() -> None:
    retrieved = ("wiki/other.md", "wiki/bayes.md", "wiki/bayes.md")
    relevant = ("wiki/bayes.md",)

    assert evaluation.recall_at_k(retrieved, relevant, k=5) == 1.0
    assert evaluation.reciprocal_rank_at_k(retrieved, relevant, k=10) == 0.5
    assert evaluation.ndcg_at_k(retrieved, relevant, k=10) == pytest.approx(1 / np.log2(3))


def test_ndcg_with_multiple_relevant_paths_is_position_sensitive() -> None:
    score = evaluation.ndcg_at_k(
        ("wiki/a.md", "wiki/no.md", "wiki/b.md"),
        ("wiki/a.md", "wiki/b.md"),
        k=3,
    )

    expected = (1.0 + 1 / np.log2(4)) / (1.0 + 1 / np.log2(3))
    assert score == pytest.approx(expected)


def test_load_cases_validates_paths_duplicates_and_requires_rows(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "id": "one",
                "slice": "study",
                "query": "bayes",
                "relevant_paths": ["wiki/bayes.md"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cases = evaluation.load_cases(cases_path)

    assert cases == (_case(),)

    cases_path.write_text(
        json.dumps(
            {
                "id": "one",
                "slice": "study",
                "query": "bayes",
                "relevant_paths": ["../private.md"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid relevant path"):
        evaluation.load_cases(cases_path)

    cases_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one"):
        evaluation.load_cases(cases_path)


def test_load_cases_can_select_one_real_data_slice(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": case_id,
                    "slice": slice_name,
                    "query": "query",
                    "relevant_paths": ["wiki/note.md"],
                }
            )
            for case_id, slice_name in (("one", "learning_notes"), ("two", "life_vault_project"))
        ),
        encoding="utf-8",
    )
    selected = evaluation.load_cases(cases_path, slice_name="life_vault_project")
    assert [case.case_id for case in selected] == ["two"]


class FakeSource:
    def search(self, query: str, limit: int = 5) -> list[KnowledgeHit]:
        document = KnowledgeDocument(
            id="wiki/bayes.md",
            path="wiki/bayes.md",
            title="Bayes",
            text="Bayes note",
        )
        return [KnowledgeHit(document=document, score=1.0, snippet=document.text)]


def test_evaluate_source_and_summarize_report_by_slice() -> None:
    samples = evaluation.evaluate_source(
        FakeSource(),
        (_case("one", "study"), _case("two", "life")),
    )

    report = evaluation.summarize(samples)

    assert report["overall"]["recall_at_5"] == 1.0
    assert sorted(report["by_slice"]) == ["life", "study"]
    assert report["by_slice"]["study"]["mrr_at_10"] == 1.0


def test_build_embedding_model_dispatches_all_four_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeModel:
        def __init__(self, name: str) -> None:
            self.name = name

    monkeypatch.setattr(
        evaluation,
        "FastEmbedModel",
        lambda: calls.append("fastembed") or FakeModel("fastembed"),
    )
    monkeypatch.setattr(
        evaluation,
        "Model2VecModel",
        lambda: calls.append("model2vec") or FakeModel("model2vec"),
    )
    monkeypatch.setattr(
        evaluation,
        "build_eval_candidate",
        lambda name: calls.append(name) or FakeModel(name),
    )

    for backend in (
        "fastembed",
        "model2vec",
        "aiteamvn_bge_m3",
        "bkai_phobert_seg",
    ):
        assert evaluation.build_embedding_model(backend).name == backend
    assert calls == [
        "fastembed",
        "model2vec",
        "aiteamvn_bge_m3",
        "bkai_phobert_seg",
    ]


def test_query_encoding_reports_mean_and_p95(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def embed_query(self, text: str) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(evaluation, "build_embedding_model", lambda backend: FakeModel())

    report = evaluation._measure_query_encoding("fastembed", (_case(),), repeats=3)

    assert report["query_mean_ms"] >= 0.0
    assert report["query_p95_ms"] >= report["query_mean_ms"]


def test_run_benchmark_chunk_sparse_separates_cold_and_warm_metrics(
    tmp_path: Path,
) -> None:
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "bayes.md").write_text("# Bayes\nBayes evidence.", encoding="utf-8")

    report = evaluation.run_benchmark(
        vault=tmp_path,
        cases=(_case(),),
        variant="chunk_sparse",
        backend="fastembed",
        rrf_k=60,
        warm_repeats=2,
    )

    assert report["status"] == "ok"
    assert report["cold"]["metrics"]["overall"]["recall_at_5"] == 1.0
    assert report["warm"]["metrics"]["overall"]["mrr_at_10"] == 1.0
    assert report["encoding"] is None
    assert "retrieved_paths" not in json.dumps(report)


def test_main_writes_unavailable_report_and_exit_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        json.dumps(
            {
                "id": "one",
                "slice": "study",
                "query": "bayes",
                "relevant_paths": ["wiki/bayes.md"],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_hybrid_retrieval",
            "--vault",
            str(tmp_path),
            "--cases",
            str(cases),
            "--variant",
            "dense",
            "--backend",
            "fastembed",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        evaluation,
        "run_benchmark",
        lambda **kwargs: (_ for _ in ()).throw(DenseUnavailableError("missing local model")),
    )

    assert evaluation.main() == 2
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "unavailable"
