from __future__ import annotations

import pytest

from scripts.smoke_test_tts import (
    VOICE_TEST_SUITES,
    build_voice_jobs,
    build_voice_texts,
    select_voices,
)


def test_select_voices_defaults_to_all_available_voices() -> None:
    assert select_voices(["NF", "SF", "NM1"], [], None) == ["NF", "SF", "NM1"]


def test_select_voices_can_use_model_specific_smoke_defaults() -> None:
    assert select_voices(
        ["auto", "female_young", "male_low"],
        [],
        None,
        default_voices=("female_young", "male_low"),
    ) == ["female_young", "male_low"]


def test_select_voices_accepts_explicit_subset() -> None:
    assert select_voices(["NF", "SF", "NM1"], ["SF"], None) == ["SF"]


def test_select_voices_can_limit_large_voice_sets_for_debugging() -> None:
    voices = [f"VIVOSSPK{i:02d}" for i in range(1, 66)]

    assert select_voices(voices, [], 3) == ["VIVOSSPK01", "VIVOSSPK02", "VIVOSSPK03"]


def test_select_voices_rejects_unknown_voice() -> None:
    with pytest.raises(ValueError, match="Unknown voice"):
        select_voices(["NF", "SF"], ["bad"], None)


def test_build_voice_texts_always_announces_voice_first() -> None:
    texts = build_voice_texts("NF", "smoke", [])

    assert texts[0] == "Đây là giọng NF."
    assert len(texts) == 3


def test_build_voice_texts_speaks_identifier_without_separators() -> None:
    texts = build_voice_texts("female_young", "smoke", [])

    assert texts[0] == "Đây là giọng female young."


def test_build_voice_texts_supports_full_suite() -> None:
    texts = build_voice_texts("NF", "full", [])

    assert texts[0] == "Đây là giọng NF."
    assert len(texts) == 3


def test_build_voice_texts_can_run_whole_suite_when_unlimited() -> None:
    texts = build_voice_texts("NF", "full", [], sentences_per_voice=0)

    assert texts[0] == "Đây là giọng NF."
    assert len(texts) == 1 + len(VOICE_TEST_SUITES["full"])


def test_custom_texts_keep_voice_intro() -> None:
    texts = build_voice_texts("SF", "full", ["Câu một.", "Câu hai.", "Câu ba."])

    assert texts == ["Đây là giọng SF.", "Câu một.", "Câu hai."]


def test_build_voice_jobs_groups_sentences_by_default() -> None:
    jobs = build_voice_jobs("female_young", "full", [])

    assert jobs == ["Đây là giọng female young. Xin chào, tôi là SoCa. Bạn muốn mình hỗ trợ gì hôm nay?"]


def test_build_voice_jobs_can_split_sentences_for_latency_debugging() -> None:
    jobs = build_voice_jobs("female_young", "full", [], split_sentences=True)

    assert jobs == [
        "Đây là giọng female young.",
        "Xin chào, tôi là SoCa.",
        "Bạn muốn mình hỗ trợ gì hôm nay?",
    ]
