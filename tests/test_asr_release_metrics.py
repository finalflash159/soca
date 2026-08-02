from __future__ import annotations

import pytest

from eval.asr_release_metrics import (
    code_switch_metrics,
    error_rates,
    ranking_metrics,
    select_lower_bound_threshold,
    wilson_interval,
)


def test_error_rates_normalize_vietnamese_and_report_counts() -> None:
    rates = error_rates(["Xin chào, THẾ GIỚI!"], ["xin chào thế giới"])

    assert rates.wer == 0.0
    assert rates.cer == 0.0
    assert rates.reference_words == 4


def test_ranking_metrics_handle_ties_without_order_bias() -> None:
    metrics = ranking_metrics(
        [True, False, True, False],
        [0.9, 0.8, 0.8, 0.1],
    )

    assert metrics.auroc == pytest.approx(0.875)
    assert metrics.average_precision == pytest.approx((1.0 + 2 / 3) / 2)


def test_code_switch_metrics_use_reference_alignment() -> None:
    metrics = code_switch_metrics(
        ["mở GitHub và chạy pytest"],
        ["mở github rồi chạy pie test"],
        [(1, 4)],
    )

    assert metrics.correct_terms == 1
    assert metrics.reference_terms == 2
    assert metrics.cs_wer == 0.5


def test_threshold_selection_honours_frozen_false_reject_budget() -> None:
    scores = [-0.9, -0.8, -0.7, -0.6, -0.5]

    assert select_lower_bound_threshold(scores, max_false_reject_rate=0.0) == -0.9
    assert select_lower_bound_threshold(scores, max_false_reject_rate=0.2) == -0.8


def test_wilson_interval_contains_observed_rate() -> None:
    lower, upper = wilson_interval(8, 10)

    assert lower < 0.8 < upper
