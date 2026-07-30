from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from local.eval_table7 import Item, run_config


class FakeRobustASR:
    def transcribe(self, _audio):
        return SimpleNamespace(
            text="nội dung sạch",
            raw_text="nội dung sạch",
            rejection_reason="",
            has_speech=True,
            was_looping=False,
            avg_logprob=-0.1,
            compression_ratio=1.0,
            total_latency_ms=12.0,
            vad=SimpleNamespace(speech_duration_ms=1000.0),
        )


class RecordingBoH:
    def __init__(self):
        self.calls = 0

    def match_and_clean(self, text):
        self.calls += 1
        return SimpleNamespace(
            cleaned_text=text.replace("nội dung", "").strip(),
            matched_phrases=("nội dung",),
        )


def _item() -> Item:
    return Item(
        audio=np.zeros(16_000, dtype=np.float32),
        ground_truth="nội dung sạch",
        duration_ms=1000.0,
        kind="speech",
    )


def test_production_no_boh_uses_unmodified_robust_asr_result() -> None:
    boh = RecordingBoH()

    result = run_config(
        "production_no_boh",
        [_item()],
        asr=SimpleNamespace(),
        vad=SimpleNamespace(),
        boh=boh,
        robust_asr=FakeRobustASR(),
    )

    assert result["predictions"] == ["nội dung sạch"]
    assert boh.calls == 0
    assert result["diagnostics"][0]["production_final_text"] == "nội dung sạch"


def test_production_with_boh_applies_only_experimental_post_processing() -> None:
    boh = RecordingBoH()

    result = run_config(
        "production_with_boh",
        [_item()],
        asr=SimpleNamespace(),
        vad=SimpleNamespace(),
        boh=boh,
        robust_asr=FakeRobustASR(),
    )

    assert result["predictions"] == ["sạch"]
    assert boh.calls == 1
    assert result["diagnostics"][0]["boh_matches"] == ["nội dung"]
