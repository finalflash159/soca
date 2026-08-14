from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.eval_tts_quality import (
    AudioItem,
    MOSModelError,
    evaluate_groups,
    load_audio_group,
    summarize_groups,
)


class _Scorer:
    revision = "fixture-revision"

    def score(self, path: Path) -> float:
        return {"candidate.wav": 3.5, "reference.wav": 4.0}[path.name]


def _item(group: str, name: str) -> AudioItem:
    return AudioItem(
        group=group,
        item_id="same-prompt",
        text_sha256="prompt-hash",
        wav_path=Path(name),
        wav_sha256=f"sha-{name}",
    )


def test_paired_relative_mos_summary_and_wer_are_kept_separate() -> None:
    rows = evaluate_groups(
        {
            "valtec": (_item("valtec", "candidate.wav"),),
            "human_reference": (_item("human_reference", "reference.wav"),),
        },
        _Scorer(),
    )
    summary = summarize_groups(
        rows,
        reference_group="human_reference",
        wer={"valtec": 0.12},
    )

    assert summary["groups"]["valtec"]["mean_mos"] == 3.5
    assert summary["groups"]["valtec"]["mean_delta_vs_reference"] == -0.5
    assert summary["groups"]["valtec"]["wer"] == 0.12
    assert summary["gate"] == {"passed": True, "reasons": []}
    assert summary["interpretation"] == "relative_vietnamese_indicator_not_absolute_mos"


def test_summary_fails_closed_without_a_human_reference_or_wer() -> None:
    rows = evaluate_groups({"valtec": (_item("valtec", "candidate.wav"),)}, _Scorer())

    summary = summarize_groups(rows, reference_group="human_reference", wer={})

    assert summary["gate"]["passed"] is False
    assert summary["gate"]["reasons"] == [
        "reference_group_missing",
        "unpaired_items:valtec",
        "wer_missing:valtec",
    ]


class _FailingScorer:
    revision = "fixture-revision"

    def score(self, path: Path) -> float:
        del path
        raise RuntimeError("boom")


def test_model_failure_is_typed_and_not_replaced_by_a_default_score() -> None:
    with pytest.raises(MOSModelError, match="prediction_failed"):
        evaluate_groups({"valtec": (_item("valtec", "candidate.wav"),)}, _FailingScorer())


def test_manifest_loader_hashes_text_and_audio(tmp_path: Path) -> None:
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"RIFF-fixture")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "records": [
                    {"item_id": "x", "text_in": "Xin chào", "wav_path": str(wav)}
                ]
            }
        ),
        encoding="utf-8",
    )

    item = load_audio_group("valtec", manifest)[0]

    assert item.group == "valtec"
    assert len(item.text_sha256) == 64
    assert len(item.wav_sha256) == 64
