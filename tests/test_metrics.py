from __future__ import annotations

import time

from soca.core.metrics import MetricsLogger


def test_metrics_stage_records_latency() -> None:
    metrics = MetricsLogger()

    with metrics.stage("asr"):
        time.sleep(0.001)

    snapshot = metrics.snapshot()
    assert "asr" in snapshot
    assert snapshot["asr"] > 0.0


def test_metrics_snapshot_returns_copy() -> None:
    metrics = MetricsLogger()

    with metrics.stage("tts"):
        pass

    snapshot = metrics.snapshot()
    snapshot["tts"] = -1.0

    assert metrics.snapshot()["tts"] >= 0.0


def test_metrics_reset_clears_recorded_stages() -> None:
    metrics = MetricsLogger()

    with metrics.stage("llm"):
        pass

    metrics.reset()

    assert metrics.snapshot() == {}


def test_metrics_stage_records_even_when_block_raises() -> None:
    metrics = MetricsLogger()

    try:
        with metrics.stage("failing"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert "failing" in metrics.snapshot()
