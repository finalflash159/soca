from __future__ import annotations

import pytest

from soca.knowledge import KnowledgeDocument, KnowledgeHit
from soca.knowledge.relevance import RelevancePolicy, assess_relevance


def _hit(
    path: str,
    title: str,
    snippet: str,
    *,
    sparse_score: float | None = None,
    dense_score: float | None = None,
    backend: str = "lexical_custom",
) -> KnowledgeHit:
    return KnowledgeHit(
        document=KnowledgeDocument(path, path, title, snippet),
        score=sparse_score or 0.0,
        snippet=snippet,
        retrieval_backend=backend,
        sparse_score=sparse_score,
        dense_score=dense_score,
    )


def test_relevance_gate_rejects_low_overlap_distractors() -> None:
    assessment = assess_relevance(
        "định lý Bayes",
        (
            _hit(
                "wiki/bayes.md",
                "Định lý Bayes",
                "Định lý Bayes cập nhật xác suất bằng bằng chứng.",
                sparse_score=100.0,
            ),
            _hit(
                "wiki/tts.md",
                "Quyết định TTS",
                "Model local giúp giảm latency khi phát giọng nói.",
                sparse_score=15.0,
            ),
        ),
    )

    assert assessment.status == "supported"
    assert [hit.document.path for hit in assessment.accepted_hits] == ["wiki/bayes.md"]
    assert assessment.rejected_count == 1


def test_dense_signal_can_admit_a_paraphrase_without_lexical_overlap() -> None:
    assessment = assess_relevance(
        "cách cập nhật niềm tin khi có quan sát mới",
        (
            _hit(
                "wiki/bayes.md",
                "Định lý Bayes",
                "Xác suất hậu nghiệm được cập nhật từ bằng chứng.",
                backend="dense",
                dense_score=0.78,
            ),
            KnowledgeHit(
                document=KnowledgeDocument("wiki/noise.md", "wiki/noise.md", "Noise", "x"),
                score=0.2,
                snippet="x",
                retrieval_backend="dense",
                dense_score=0.31,
            ),
        ),
    )

    assert assessment.status == "supported"
    assert [hit.document.path for hit in assessment.accepted_hits] == ["wiki/bayes.md"]


def test_dense_margin_keeps_near_tied_runner_up_below_admission_floor() -> None:
    assessment = assess_relevance(
        "tôi ghi chú gì trong deeplearning vậy",
        (
            _hit(
                "wiki/life/journal.md",
                "Nhật ký",
                "Cách tôi muốn viết ghi chú học tập.",
                backend="dense",
                dense_score=0.5547,
            ),
            _hit(
                "wiki/learning/deep-learning/attention.md",
                "Attention và Transformer",
                "Attention trộn thông tin giữa các token.",
                backend="dense",
                dense_score=0.5462,
            ),
        ),
    )

    assert assessment.status == "weak"
    assert assessment.reason == "ambiguous_top_margin"
    assert assessment.margin == pytest.approx(0.0085)
    assert [hit.document.path for hit in assessment.accepted_hits] == ["wiki/life/journal.md"]


def test_relevance_policy_is_calibratable_without_code_changes() -> None:
    policy = RelevancePolicy(min_lexical_coverage=0.9)
    assessment = assess_relevance(
        "truy hồi chủ đề không tồn tại",
        (
            _hit(
                "wiki/bayes.md",
                "Định lý Bayes",
                "Bayes cập nhật xác suất.",
                sparse_score=10.0,
            ),
        ),
        policy=policy,
    )

    assert assessment.status == "insufficient"
    assert assessment.reason == "all_hits_below_floor"


def test_retrieval_modes_keep_separate_score_distributions() -> None:
    sparse = RelevancePolicy.for_retrieval_mode("cached_sparse")
    hybrid = RelevancePolicy.for_retrieval_mode("hybrid")

    assert sparse.min_lexical_coverage == 0.65
    assert sparse.min_dense_score == 0.55
    assert hybrid.min_lexical_coverage == 0.95
    assert hybrid.min_dense_score == 0.52


def test_hybrid_does_not_admit_sparse_rank_without_query_coverage() -> None:
    assessment = assess_relevance(
        "lượng tử topological superconductors",
        (
            KnowledgeHit(
                document=KnowledgeDocument(
                    "wiki/tts.md",
                    "wiki/tts.md",
                    "Quyết định TTS",
                    "SoCa chọn Valtec ONNX làm baseline.",
                ),
                score=0.8,
                snippet="SoCa chọn Valtec ONNX làm baseline.",
                retrieval_backend="hybrid",
                sparse_score=0.8,
                dense_score=0.40,
                fusion_score=0.8,
            ),
        ),
        policy=RelevancePolicy.for_retrieval_mode("hybrid"),
    )

    assert assessment.status == "insufficient"
    assert assessment.accepted_hits == ()


def test_generic_lexical_overlap_is_not_enough_when_sparse_score_is_weak() -> None:
    assessment = assess_relevance(
        "hệ thống hoạt động hiệu quả",
        (
            _hit(
                "wiki/irrelevant.md",
                "Hệ thống",
                "Một hệ thống hoạt động theo các bước chung.",
                sparse_score=10.0,
            ),
            _hit(
                "wiki/answer.md",
                "Khác chủ đề",
                "Hệ thống hoạt động hiệu quả khi các bước được kiểm tra.",
                sparse_score=100.0,
            ),
        ),
        policy=RelevancePolicy(min_lexical_coverage=0.9, min_sparse_score_ratio=0.75),
    )

    assert assessment.status == "supported"
    assert [hit.document.path for hit in assessment.accepted_hits] == ["wiki/answer.md"]
    assert assessment.rejected_count == 1


def test_relevance_gate_preserves_order_across_backend_score_spaces() -> None:
    assessment = assess_relevance(
        "định lý Bayes",
        (
            _hit(
                "wiki/lexical.md",
                "Bayes lexical",
                "Định lý Bayes cập nhật xác suất.",
                sparse_score=70.0,
            ),
            _hit(
                "wiki/dense.md",
                "Bayes paraphrase",
                "Cập nhật niềm tin sau quan sát mới.",
                backend="dense",
                dense_score=0.91,
            ),
        ),
    )

    assert [hit.document.path for hit in assessment.accepted_hits] == [
        "wiki/lexical.md",
        "wiki/dense.md",
    ]
    assert assessment.margin is None


def test_hybrid_margin_does_not_compare_dense_and_sparse_signals() -> None:
    assessment = assess_relevance(
        "định lý Bayes",
        (
            _hit(
                "wiki/dense.md",
                "Bayes dense",
                "Cập nhật niềm tin sau quan sát mới.",
                backend="hybrid",
                dense_score=0.91,
            ),
            _hit(
                "wiki/sparse.md",
                "Bayes sparse",
                "Định lý Bayes cập nhật xác suất.",
                backend="hybrid",
                sparse_score=80.0,
            ),
        ),
    )

    assert assessment.margin is None


def test_same_backend_margin_is_sorted_before_comparison() -> None:
    assessment = assess_relevance(
        "định lý Bayes",
        (
            _hit(
                "wiki/later.md",
                "Bayes later",
                "Bayes evidence.",
                backend="dense",
                dense_score=0.60,
            ),
            _hit(
                "wiki/top.md",
                "Bayes top",
                "Bayes evidence.",
                backend="dense",
                dense_score=0.80,
            ),
        ),
    )

    assert assessment.margin == pytest.approx(0.2)


def test_fusion_floor_rejects_lower_ranked_distractors() -> None:
    assessment = assess_relevance(
        "định lý Bayes",
        (
            KnowledgeHit(
                document=KnowledgeDocument(
                    "wiki/bayes.md",
                    "wiki/bayes.md",
                    "Định lý Bayes",
                    "Định lý Bayes evidence.",
                ),
                score=1.0,
                snippet="Định lý Bayes evidence.",
                retrieval_backend="hybrid",
                sparse_score=10.0,
                fusion_score=1.0,
            ),
            KnowledgeHit(
                document=KnowledgeDocument("wiki/noise.md", "wiki/noise.md", "Noise", "x"),
                score=0.2,
                snippet="x",
                retrieval_backend="hybrid",
                dense_score=0.8,
                fusion_score=0.2,
            ),
        ),
    )

    assert [hit.document.path for hit in assessment.accepted_hits] == ["wiki/bayes.md"]
