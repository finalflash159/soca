from __future__ import annotations

import pytest

from scripts.smoke_test_tts import (
    VOICE_TEST_SUITES,
    build_parser,
    build_voice_jobs,
    build_voice_texts,
    select_voices,
)


def test_parser_rejects_tts_model_selector() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--model", "other_tts"])


def test_select_voices_defaults_to_all_available_voices() -> None:
    assert select_voices(["NF", "SF", "NM1"], [], None) == ["NF", "SF", "NM1"]


def test_select_voices_accepts_explicit_subset() -> None:
    assert select_voices(["NF", "SF", "NM1"], ["SF"], None) == ["SF"]


def test_select_voices_can_limit_valtec_voice_set_for_debugging() -> None:
    voices = ["NF", "SF", "NM1", "SM", "NM2"]

    assert select_voices(voices, [], 3) == ["NF", "SF", "NM1"]


def test_select_voices_rejects_unknown_voice() -> None:
    with pytest.raises(ValueError, match="Unknown voice"):
        select_voices(["NF", "SF"], ["bad"], None)


def test_build_voice_texts_always_announces_voice_first() -> None:
    texts = build_voice_texts("NF", "smoke", [])

    assert texts[0] == "Đây là giọng NF."
    assert len(texts) == 3


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
    jobs = build_voice_jobs("NF", "full", [])

    assert jobs == ["Đây là giọng NF. Xin chào, tôi là SoCa. Bạn muốn mình hỗ trợ gì hôm nay?"]


def test_build_voice_jobs_can_split_sentences_for_latency_debugging() -> None:
    jobs = build_voice_jobs("NF", "full", [], split_sentences=True)

    assert jobs == [
        "Đây là giọng NF.",
        "Xin chào, tôi là SoCa.",
        "Bạn muốn mình hỗ trợ gì hôm nay?",
    ]
