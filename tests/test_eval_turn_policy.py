from __future__ import annotations

import json
from pathlib import Path

from eval.eval_turn_policy import evaluate


def test_policy_evaluator_separates_unsupported_tool_and_parity(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps({"id": "weather-text", "family": "weather", "query": "thời tiết", "disposition": "out_of_scope", "sources": []}),
                json.dumps({"id": "weather-asr", "family": "weather", "query": "thời tiết", "disposition": "out_of_scope", "sources": []}),
            ]
        ) + "\n",
        encoding="utf-8",
    )
    predictions.write_text(
        "\n".join(
            [
                json.dumps({"id": "weather-text", "disposition": "out_of_scope", "sources": [], "tool": "none", "parity_id": "weather"}),
                json.dumps({"id": "weather-asr", "disposition": "direct_tool", "sources": [], "tool": "local_time.now", "parity_id": "weather"}),
            ]
        ) + "\n",
        encoding="utf-8",
    )
    result = evaluate(dataset, predictions)
    assert result["unsupported_to_real_tool"] == {"count": 1, "denominator": 2, "rate": 0.5}
    assert result["chat_voice_parity"]["mismatch_count"] == 1
