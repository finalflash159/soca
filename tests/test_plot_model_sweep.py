"""Smoke tests for the research-only cross-model ASR charts."""

from __future__ import annotations

import json
from pathlib import Path

from local.plot_model_sweep import load_sweep, render_sweep


def _fake_report(
    wer: float,
    halluc_raw: float,
    halluc_experimental: float,
    dur_ms: float,
    lat_ms: float,
) -> dict:
    return {
        "results": {
            "raw": {
                "wer": wer,
                "cer": wer / 2,
                "hallucination_rate": halluc_raw,
                "latency_mean_ms": lat_ms,
                "diagnostics": [{"speech_duration_ms": dur_ms}],
            },
            "vad_deloop_boh": {"hallucination_rate": halluc_experimental},
        }
    }


def test_render_sweep_writes_two_charts(tmp_path: Path) -> None:
    rows = [
        {
            "name": "tiny",
            "params_m": 39,
            "wer_raw": 20.5,
            "cer_raw": 9.4,
            "halluc_raw": 100.0,
            "halluc_experimental": 3.3,
            "rtf": 0.12,
        },
        {
            "name": "large",
            "params_m": 1550,
            "wer_raw": 12.4,
            "cer_raw": 5.5,
            "halluc_raw": 100.0,
            "halluc_experimental": 10.0,
            "rtf": 2.86,
        },
    ]
    written = render_sweep(rows, tmp_path)

    names = {p.name for p in written}
    assert names == {"model_sweep_wer_rtf.png", "model_sweep_halluc.png"}
    for path in written:
        assert path.exists() and path.stat().st_size > 0


def test_load_sweep_computes_rtf_and_skips_missing(tmp_path: Path) -> None:
    # tiny present (dur 10s, lat 1.2s -> RTF 0.12); "base" file absent -> skipped.
    (tmp_path / "table7_phowhisper_tiny_focused.json").write_text(
        json.dumps(_fake_report(0.205, 1.0, 0.033, 10000, 1200)), encoding="utf-8"
    )
    sweep = [
        ("tiny", 39, "table7_phowhisper_tiny_focused.json"),
        ("base", 74, "table7_phowhisper_base.json"),
    ]
    rows = load_sweep(tmp_path, sweep)

    assert [r["name"] for r in rows] == ["tiny"]  # missing base skipped
    assert rows[0]["wer_raw"] == 20.5
    assert abs(rows[0]["rtf"] - 0.12) < 1e-9


def test_load_sweep_falls_back_to_second_candidate(tmp_path: Path) -> None:
    # Only the fallback file (table7_replication.json) is present; the primary
    # focused name is absent → tiny is still picked up, not skipped.
    (tmp_path / "table7_replication.json").write_text(
        json.dumps(_fake_report(0.205, 1.0, 0.033, 10000, 1200)), encoding="utf-8"
    )
    sweep = [
        ("tiny", 39, ("table7_phowhisper_tiny_focused.json", "table7_replication.json")),
    ]
    rows = load_sweep(tmp_path, sweep)

    assert [r["name"] for r in rows] == ["tiny"]
    assert rows[0]["wer_raw"] == 20.5
