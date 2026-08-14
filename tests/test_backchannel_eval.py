from __future__ import annotations

from eval.eval_backchannel_classifier import ClassificationReceipt, evaluate_receipts


def test_backchannel_gate_scores_both_classes_and_latency() -> None:
    receipts = tuple(
        [
            ClassificationReceipt(
                case_id=f"b-{index}",
                expected="backchannel",
                predicted="backchannel",
                confidence=0.95,
                latency_ms=100.0,
                model="easy-turn",
                revision="abc",
                audio_sha256=f"{index:064x}",
            )
            for index in range(10)
        ]
        + [
            ClassificationReceipt(
                case_id=f"i-{index}",
                expected="interruption",
                predicted="interruption",
                confidence=0.96,
                latency_ms=120.0,
                model="easy-turn",
                revision="abc",
                audio_sha256=f"{index + 20:064x}",
            )
            for index in range(10)
        ]
    )

    report = evaluate_receipts(receipts)

    assert report["gate_status"] == "pass"
    assert report["backchannel_recall"] == 1.0
    assert report["interruption_recall"] == 1.0
    assert report["latency_p95_ms"] == 120.0


def test_gate_fails_closed_on_model_drift_or_missing_class() -> None:
    receipts = (
        ClassificationReceipt(
            case_id="b-1",
            expected="backchannel",
            predicted="backchannel",
            confidence=0.9,
            latency_ms=10.0,
            model="one",
            revision="r1",
            audio_sha256="a" * 64,
        ),
        ClassificationReceipt(
            case_id="i-1",
            expected="interruption",
            predicted="interruption",
            confidence=0.9,
            latency_ms=10.0,
            model="two",
            revision="r2",
            audio_sha256="b" * 64,
        ),
    )

    report = evaluate_receipts(receipts)

    assert report["gate_status"] == "fail"
    assert "model_identity_drift" in report["failures"]
