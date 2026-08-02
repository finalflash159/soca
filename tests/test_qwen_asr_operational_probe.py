from __future__ import annotations

from eval.qwen_asr_operational_probe import _summarize


def test_operational_summary_counts_cleanup_and_fallback_receipts() -> None:
    payload = {
        "start_stop": [
            {
                "startup_ms": 10.0,
                "exit_code": 0,
                "socket_removed": True,
                "ready_marker_removed": True,
                "no_fallback_attempted": True,
                "cold_inference": {"wall_ms": 20.0, "ipc_overhead_ms": 1.0},
            },
            {
                "startup_ms": 12.0,
                "exit_code": 0,
                "socket_removed": False,
                "ready_marker_removed": True,
                "no_fallback_attempted": False,
                "cold_inference": {"wall_ms": 22.0, "ipc_overhead_ms": 2.0},
            },
        ],
        "warm": {
            "short": {
                "partials": [
                    {"fraction": 0.25, "repetition": 0, "wall_ms": 3.0, "text": "a"},
                    {"fraction": 0.5, "repetition": 0, "wall_ms": 4.0, "text": "a b"},
                    {"fraction": 0.75, "repetition": 0, "wall_ms": 5.0, "text": "a b c"},
                ]
            }
        },
    }

    summary = _summarize(payload)

    assert summary["orphan_process_count"] == 1
    assert summary["fallback_attempt_count"] == 1
