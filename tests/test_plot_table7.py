"""Smoke tests for research-only ASR chart rendering."""

from __future__ import annotations

from pathlib import Path

from local.plot_table7 import render_all


def _fake_report() -> dict:
    def cfg_result(wer: float, halluc: float, frej: float, breakdown: dict, by_sub: dict) -> dict:
        return {
            "wer": wer,
            "hallucination_rate": halluc,
            "robustness": {
                "false_reject_rate": frej,
                "n_noise": 100,
                "noise_stage_breakdown": breakdown,
                "hallucination_rate_by_subtype": by_sub,
            },
        }

    return {
        "metadata": {"asr_runtime_identity": {"model_key": "phowhisper_tiny"}},
        "results": {
            "raw": cfg_result(0.12, 1.0, 0.0, {"accepted": 100}, {"pure": 1.0, "speech_like": 1.0}),
            "vad_deloop_boh": cfg_result(
                0.14, 0.03, 0.02,
                {"no_speech": 80, "low_confidence": 12, "heuristic": 5, "accepted": 3},
                {"pure": 0.0, "speech_like": 0.15},
            ),
        },
    }


def test_render_all_writes_expected_charts(tmp_path: Path) -> None:
    written = render_all(_fake_report(), tmp_path)

    names = {p.name for p in written}
    assert names == {
        "wer_vs_halluc_phowhisper_tiny.png",
        "stage_contribution_phowhisper_tiny.png",
        "halluc_by_subtype_phowhisper_tiny.png",
    }
    for path in written:
        assert path.exists()
        assert path.stat().st_size > 0


def test_render_all_skips_stage_chart_without_experimental_config(tmp_path: Path) -> None:
    report = _fake_report()
    del report["results"]["vad_deloop_boh"]

    written = render_all(report, tmp_path)

    names = {p.name for p in written}
    assert "stage_contribution_phowhisper_tiny.png" not in names
    assert "wer_vs_halluc_phowhisper_tiny.png" in names
