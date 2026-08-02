from __future__ import annotations

from dataclasses import dataclass

from soca.tts.valtec.lexicon import ACRONYM_LEXICON, WORD_LEXICON

from .scoring import MatchMode


@dataclass(frozen=True)
class CorpusItem:
    item_id: str
    corpus: str
    text_in: str
    expected: str
    mode: MatchMode


# Input -> spoken form, lifted from tests/test_valtec_normalizer.py so the
# audio-level check asserts the same expectations the text-level tests already
# lock in. Keeping one source of truth means a normalizer change cannot pass
# one layer and silently break the other.
_NORMALIZER_CASES: tuple[tuple[str, str], ...] = (
    ("Giá là 100.000đ", "một trăm nghìn đồng"),
    ("Giá là 13.800,25 USD", "mười ba nghìn, tám trăm phẩy hai năm đô la"),
    ("Họp lúc 14:30", "mười bốn giờ ba mươi phút"),
    ("Họp lúc 15h30", "mười lăm giờ ba mươi phút"),
    ("Khoảng 2 giờ 20 phút", "hai giờ hai mươi phút"),
    ("Bắt đầu lúc 2 giờ", "hai giờ"),
    ("Sinh ngày 15/08/1990", "ngày mười lăm tháng tám năm"),
    ("Nhuận ngày 29/02/2024", "ngày hai mươi chín tháng hai năm"),
    ("Hẹn ngày 15", "ngày mười lăm"),
    ("Kế hoạch tháng 8", "tháng tám"),
    ("Hẹn 8 tháng 11", "ngày tám tháng mười một"),
    ("Tỷ lệ 85%", "tám mươi lăm phần trăm"),
    ("Nhiệt độ 25.5 độ C", "hai mươi lăm phẩy năm"),
    ("Sai số 1,05", "một phẩy không năm"),
    ("Số điện thoại 0912345678", "không chín một hai"),
    ("Gọi +84 912 345 678", "không chín một hai ba bốn năm sáu bảy tám"),
)

# Plain Vietnamese with no technical vocabulary and no numbers. Both engines
# must score near-perfect here; a bad score is evidence against the measuring
# chain (TTS voice, resampling, ASR) rather than against the term under test,
# which is the only way to tell a real defect from a noisy round trip.
_CONTROL_SENTENCES: tuple[str, ...] = (
    "Hôm nay trời khá đẹp và mát mẻ",
    "Tôi muốn uống một cốc cà phê nóng",
    "Bạn có thể nói chậm lại một chút không",
    "Buổi họp sáng nay diễn ra rất suôn sẻ",
    "Chiều mai tôi sẽ ghé qua nhà bạn chơi",
    "Con mèo đang nằm ngủ trên chiếc ghế gỗ",
    "Cảm ơn bạn đã giúp tôi hoàn thành việc này",
    "Chúng ta cùng đi ăn tối ở quán quen nhé",
)

# A carrier sentence gives the TTS normal prosodic context and the ASR normal
# language-model context. A bare term measures neither engine in the way they
# are actually used.
_CARRIER = "Mô hình dùng {term} để so sánh kết quả"
_ACRONYM_CARRIER = "Tài liệu này nói về {term} rất chi tiết"


def normalizer_corpus() -> tuple[CorpusItem, ...]:
    return tuple(
        CorpusItem(
            item_id=f"normalizer-{index:03d}",
            corpus="normalizer",
            text_in=text_in,
            expected=expected,
            mode="contains",
        )
        for index, (text_in, expected) in enumerate(_NORMALIZER_CASES)
    )


def lexicon_corpus(limit: int | None = None) -> tuple[CorpusItem, ...]:
    """Curated technical terms, each in a carrier sentence.

    These entries exist in the Valtec lexicon precisely because statistical G2P
    got them wrong, so they are the terms most likely to expose a frontend
    regression -- the highest-signal sample available without new labelling.
    """
    words = sorted(WORD_LEXICON)
    acronyms = sorted(ACRONYM_LEXICON)

    items: list[CorpusItem] = []
    for index, term in enumerate(words):
        items.append(
            CorpusItem(
                item_id=f"lexicon-word-{index:03d}",
                corpus="lexicon",
                text_in=_CARRIER.format(term=term),
                expected=term,
                mode="term",
            )
        )
    for index, term in enumerate(acronyms):
        items.append(
            CorpusItem(
                item_id=f"lexicon-acronym-{index:03d}",
                corpus="lexicon",
                text_in=_ACRONYM_CARRIER.format(term=term),
                expected=term,
                mode="term",
            )
        )
    return tuple(items[:limit] if limit is not None else items)


def control_corpus() -> tuple[CorpusItem, ...]:
    return tuple(
        CorpusItem(
            item_id=f"control-{index:03d}",
            corpus="control",
            text_in=sentence,
            expected=sentence,
            mode="exact",
        )
        for index, sentence in enumerate(_CONTROL_SENTENCES)
    )


def build_all_corpora(*, lexicon_limit: int | None = None) -> dict[str, tuple[CorpusItem, ...]]:
    return {
        "normalizer": normalizer_corpus(),
        "lexicon": lexicon_corpus(limit=lexicon_limit),
        "control": control_corpus(),
    }


__all__ = [
    "CorpusItem",
    "build_all_corpora",
    "control_corpus",
    "lexicon_corpus",
    "normalizer_corpus",
]
