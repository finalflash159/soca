from soca.core.streaming import pop_ready_sentence


def test_pop_ready_sentence_waits_for_min_chars():
    sentence, rest = pop_ready_sentence("OK.", min_chars=10)

    assert sentence is None
    assert rest == "OK."


def test_pop_ready_sentence_returns_first_complete_sentence():
    sentence, rest = pop_ready_sentence("Xin chào bạn. Hôm nay trời đẹp.", min_chars=10)

    assert sentence == "Xin chào bạn."
    assert rest == "Hôm nay trời đẹp."


def test_pop_ready_sentence_merges_short_greeting_with_next_sentence():
    sentence, rest = pop_ready_sentence(
        "Xin chào! Tôi là SoCa. Tôi có thể giúp gì cho bạn?",
        min_chars=24,
    )

    assert sentence == "Xin chào! Tôi là SoCa. Tôi có thể giúp gì cho bạn?"
    assert rest == ""


def test_pop_ready_sentence_waits_when_no_sentence_boundary():
    sentence, rest = pop_ready_sentence("Đây là một câu chưa có dấu kết", min_chars=10)

    assert sentence is None
    assert rest == "Đây là một câu chưa có dấu kết"


def test_pop_ready_sentence_does_not_emit_bare_numbered_list_marker():
    buffer = "Tình hình:\n1. "

    sentence, rest = pop_ready_sentence(buffer, min_chars=5)

    assert sentence is None
    assert rest == buffer


def test_pop_ready_sentence_keeps_numbered_marker_with_item_text():
    sentence, rest = pop_ready_sentence(
        "Tình hình:\n1. Ăn đủ đạm. 2. Ngủ sớm.",
        min_chars=5,
    )

    assert sentence == "Tình hình:\n1. Ăn đủ đạm."
    assert rest == "2. Ngủ sớm."
