# tests/test_valtec_normalizer.py
import re
import unicodedata

import pytest

from soca.tts.valtec.normalizer import ValtecTextNormalizer, number_to_words


@pytest.mark.parametrize(
    ("raw", "expected_contains"),
    [
        ("Giá là 100.000đ", "một trăm nghìn đồng"),
        ("Giá là 13.800,25 USD", "mười ba nghìn tám trăm phẩy hai năm đô la"),
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
    ],
)
def test_valtec_normalizer(raw, expected_contains):
    assert expected_contains in ValtecTextNormalizer().normalize(raw)


def test_normalizer_is_nfc_and_idempotent():
    normalizer = ValtecTextNormalizer()
    decomposed = unicodedata.normalize("NFD", "Đặng Thùy Trâm")
    once = normalizer.normalize(decomposed)
    assert unicodedata.is_normalized("NFC", once)
    assert normalizer.normalize(once) == once


@pytest.mark.parametrize("raw", ["99/99/2026", "29/02/2023", "31/04/2026"])
def test_invalid_date_is_not_presented_as_valid_date(raw):
    result = ValtecTextNormalizer().normalize(raw)
    assert "ngày" not in result
    assert "tháng" not in result


def test_email_and_url_are_removed_before_symbol_expansion():
    result = ValtecTextNormalizer().normalize("Gửi a@example.com hoặc https://example.com nhé")
    assert "example" not in result
    assert "a còng" not in result


@pytest.mark.parametrize(
    ("raw", "spoken"),
    [
        ("0", "không"),
        ("15", "mười lăm"),
        ("21", "hai mươi mốt"),
        ("24", "hai mươi tư"),
        ("25", "hai mươi lăm"),
        ("100", "một trăm"),
        ("101", "một trăm lẻ một"),
        ("105", "một trăm lẻ năm"),
        ("115", "một trăm mười lăm"),
        ("121", "một trăm hai mươi mốt"),
        ("1005", "một nghìn lẻ năm"),
        ("1000001", "một triệu lẻ một"),
        ("1001001", "một triệu một nghìn lẻ một"),
        ("1000000001", "một tỷ lẻ một"),
        ("1000000000001", "một nghìn tỷ lẻ một"),
        ("-105", "âm một trăm lẻ năm"),
    ],
)
def test_number_to_words(raw, spoken):
    assert number_to_words(raw) == spoken


def test_number_to_words_preserves_invalid_input_for_caller_policy():
    assert number_to_words("12A3") == "12A3"


def test_number_to_words_never_leaves_digits_or_double_spaces():
    values = [*range(2000), 10**6 + 1, 10**9 + 1, 10**12 + 1, 10**30 + 1]
    for value in values:
        spoken = number_to_words(str(value))
        assert re.search(r"\d", spoken) is None, (value, spoken)
        assert "  " not in spoken, (value, spoken)