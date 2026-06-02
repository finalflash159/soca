from __future__ import annotations

import json
from pathlib import Path

from scripts.audition_omnivoice_voice import DEFAULT_REFERENCE_TEXT, select_voices
from scripts.freeze_omnivoice_voice import find_sample_row


def test_audition_selects_default_design_voices() -> None:
    voices = select_voices(
        ["auto", "female_young", "male_low"],
        [],
        ("female_young", "male_low"),
        max_voices=None,
    )

    assert voices == ["female_young", "male_low"]


def test_reference_text_is_long_enough_for_voice_clone_prompt() -> None:
    assert len(DEFAULT_REFERENCE_TEXT.split()) >= 25


def test_freeze_finds_sample_text_from_manifest(tmp_path: Path) -> None:
    sample = tmp_path / "female_young" / "female_young_01.wav"
    sample.parent.mkdir()
    sample.write_bytes(b"fake wav")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audio_path": str(sample),
                "voice": "female_young",
                "text": "Xin chào, tôi là SoCa.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    row = find_sample_row(sample, manifest)

    assert row is not None
    assert row["text"] == "Xin chào, tôi là SoCa."
