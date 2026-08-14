from __future__ import annotations

from dataclasses import replace

from eval.eval_speculative_retrieval import (
    SpeculativeLatencyReceipt,
    evaluate_receipts,
)


def _receipt(index: int) -> SpeculativeLatencyReceipt:
    result_hash = f"{index + 100:064x}"
    return SpeculativeLatencyReceipt(
        turn_id=f"turn-{index}",
        query_sha256=f"{index:064x}",
        source_identity="generation-a",
        baseline_result_sha256=result_hash,
        speculative_result_sha256=result_hash,
        baseline_visible_ms=70.0 + index,
        speculative_visible_ms=1.0 + index / 100,
        prefetch_lead_ms=100.0,
        cache_status="hit" if index < 18 else "miss",
        controller_terminal="achieved",
        verification_passed=True,
    )


def test_speculative_latency_gate_requires_equivalent_verified_results() -> None:
    report = evaluate_receipts(tuple(_receipt(index) for index in range(20)))

    assert report["gate_status"] == "pass"
    assert report["cache_hit_rate"] == 0.9
    assert report["median_visible_latency_saved_ms"] > 0


def test_speculative_latency_gate_rejects_result_or_terminal_drift() -> None:
    receipts = [_receipt(index) for index in range(20)]
    receipts[0] = replace(
        receipts[0],
        speculative_result_sha256="f" * 64,
        controller_terminal="system_failure",
    )

    report = evaluate_receipts(tuple(receipts))

    assert report["gate_status"] == "fail"
    assert "result_mismatch" in report["reasons"]
    assert "controller_or_verification_failed" in report["reasons"]
