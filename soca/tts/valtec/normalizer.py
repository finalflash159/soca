from __future__ import annotations

import re
import unicodedata
from datetime import date

DIGITS = {
    "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
    "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín",
}
GROUP_PREFIXES = ("", "nghìn", "triệu")
ORDINALS = {"1": "nhất", "2": "hai", "3": "ba", "4": "tư", "5": "năm"}


def _read_triplet(value: int) -> str:
    if not 0 < value < 1000:
        raise ValueError("triplet must be in range 1..999")
    hundreds, remainder = divmod(value, 100)
    tens, units = divmod(remainder, 10)
    words: list[str] = []
    if hundreds:
        words.extend((DIGITS[str(hundreds)], "trăm"))
    if tens >= 2:
        words.extend((DIGITS[str(tens)], "mươi"))
        if units == 1:
            words.append("mốt")
        elif units == 4:
            words.append("tư")
        elif units == 5:
            words.append("lăm")
        elif units:
            words.append(DIGITS[str(units)])
    elif tens == 1:
        words.append("mười")
        if units == 5:
            words.append("lăm")
        elif units:
            words.append(DIGITS[str(units)])
    elif units:
        if hundreds:
            words.append("lẻ")
        words.append(DIGITS[str(units)])
    return " ".join(words)


def _group_name(index: int) -> str:
    """Return nghìn/triệu/tỷ groups without imposing a 12-digit ceiling."""
    if index == 0:
        return ""
    ty_count, prefix_index = divmod(index, 3)
    return " ".join(
        part for part in (GROUP_PREFIXES[prefix_index], " ".join(["tỷ"] * ty_count)) if part
    )


def number_to_words(raw: str) -> str:
    candidate = raw.strip()
    if re.fullmatch(r"[+-]?\d+", candidate) is None:
        return raw
    negative = candidate.startswith("-")
    digits = candidate.lstrip("+-").lstrip("0") or "0"
    value = int(digits)
    if value == 0:
        return DIGITS["0"]
    groups: list[int] = []
    while value:
        value, group = divmod(value, 1000)
        groups.append(group)
    words: list[str] = []
    for index in range(len(groups) - 1, -1, -1):
        group = groups[index]
        if group == 0:
            continue
        # Natural TTS reading uses one connector before a final 1..99 remainder:
        # 1_000_001 -> "một triệu lẻ một", not "một triệu không trăm lẻ một".
        if words and index == 0 and group < 100:
            words.append("lẻ")
        words.append(_read_triplet(group))
        group_name = _group_name(index)
        if group_name:
            words.append(group_name)
    spoken = " ".join(words)
    return f"âm {spoken}" if negative else spoken


def _split_formatted_number(raw: str) -> tuple[str, str | None]:
    """Split a Vietnamese/US formatted token into integer and fractional digits."""
    sign = "-" if raw.startswith("-") else ""
    token = raw.lstrip("+-")
    if not token or re.fullmatch(r"\d+(?:[.,]\d+)*", token) is None:
        raise ValueError(f"invalid numeric token: {raw!r}")

    # A repeated, consistent group of three is an integer thousands form.
    if re.fullmatch(r"\d{1,3}(?:([.,])\d{3})(?:\1\d{3})*", token):
        return sign + re.sub(r"[.,]", "", token), None

    separators = [index for index, char in enumerate(token) if char in ".,"]
    if not separators:
        return sign + token, None
    last_separator = separators[-1]
    integer_raw, fractional = token[:last_separator], token[last_separator + 1 :]

    # With both separators, the last one is decimal and earlier ones are grouping.
    # With one separator, one/two trailing digits are decimal; three are grouping above.
    if len(separators) == 1 and len(fractional) > 2:
        raise ValueError(f"ambiguous numeric token: {raw!r}")
    integer = re.sub(r"[.,]", "", integer_raw)
    return sign + integer, fractional


def _spoken_numeric(raw: str) -> str:
    try:
        integer, fractional = _split_formatted_number(raw)
    except ValueError:
        return raw
    spoken = number_to_words(integer)
    if fractional is None:
        return spoken
    # Read fractional digits individually so 1.05 keeps the semantically required zero.
    return f"{spoken} phẩy {' '.join(DIGITS[digit] for digit in fractional)}"


def _convert_dates(text: str) -> str:
    full_pattern = re.compile(
        r"(?:\bngày\s+)?(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", re.IGNORECASE
    )

    def replace_full(match: re.Match[str]) -> str:
        day, month, year = (int(match.group(index)) for index in (1, 2, 3))
        try:
            date(year, month, day)
        except ValueError:
            return match.group(0)
        return (
            f"ngày {number_to_words(str(day))} tháng {number_to_words(str(month))} "
            f"năm {number_to_words(str(year))}"
        )

    text = full_pattern.sub(replace_full, text)
    words_pattern = re.compile(
        r"(?:\bngày\s+)?(\d{1,2})\s+tháng\s+(\d{1,2})"
        r"(?:\s+năm\s+(\d{4}))?\b",
        re.IGNORECASE,
    )

    def replace_words(match: re.Match[str]) -> str:
        day, month = int(match.group(1)), int(match.group(2))
        year = int(match.group(3)) if match.group(3) else None
        try:
            # Leap year 2000 allows 29/02 when the input omits a year.
            date(year or 2000, month, day)
        except ValueError:
            return match.group(0)
        result = f"ngày {number_to_words(str(day))} tháng {number_to_words(str(month))}"
        if year is not None:
            result += f" năm {number_to_words(str(year))}"
        return result

    text = words_pattern.sub(replace_words, text)

    def replace_day(match: re.Match[str]) -> str:
        day = int(match.group(1))
        return f"ngày {number_to_words(str(day))}" if 1 <= day <= 31 else match.group(0)

    def replace_month(match: re.Match[str]) -> str:
        month = int(match.group(1))
        return f"tháng {number_to_words(str(month))}" if 1 <= month <= 12 else match.group(0)

    text = re.sub(r"\bngày\s+(\d{1,2})\b", replace_day, text, flags=re.IGNORECASE)
    return re.sub(r"\btháng\s+(\d{1,2})\b", replace_month, text, flags=re.IGNORECASE)


def _convert_times(text: str) -> str:
    pattern = re.compile(r"\b(\d{1,2})(?::|h)(\d{2})(?::(\d{2}))?\b", re.IGNORECASE)

    def replace(match: re.Match[str]) -> str:
        hour = int(match.group(1))
        minute = int(match.group(2))
        second = int(match.group(3)) if match.group(3) else None
        if hour > 23 or minute > 59 or (second is not None and second > 59):
            return match.group(0)
        result = f"{number_to_words(str(hour))} giờ {number_to_words(str(minute))} phút"
        if second is not None:
            result += f" {number_to_words(str(second))} giây"
        return result

    text = pattern.sub(replace, text)
    words_pattern = re.compile(
        r"\b(\d{1,2})\s*giờ\s*(\d{1,2})\s*phút"
        r"(?:\s*(\d{1,2})\s*giây)?\b",
        re.IGNORECASE,
    )
    text = words_pattern.sub(replace, text)

    def replace_hour(match: re.Match[str]) -> str:
        hour = int(match.group(1))
        return f"{number_to_words(str(hour))} giờ" if hour <= 23 else match.group(0)

    return re.sub(r"\b(\d{1,2})\s*giờ\b", replace_hour, text, flags=re.IGNORECASE)


def _convert_currency(text: str) -> str:
    numeric = r"(\d+(?:[.,]\d+)*)"
    text = re.sub(
        numeric + r"\s*(?:VND|VNĐ|đồng)\b",
        lambda match: f"{_spoken_numeric(match.group(1))} đồng",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        numeric + r"\s*đ(?![A-Za-zÀ-ỹ])",
        lambda match: f"{_spoken_numeric(match.group(1))} đồng",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\$\s*" + numeric,
        lambda match: f"{_spoken_numeric(match.group(1))} đô la",
        text,
    )
    return re.sub(
        numeric + r"\s*(?:USD|\$)",
        lambda match: f"{_spoken_numeric(match.group(1))} đô la",
        text,
        flags=re.IGNORECASE,
    )


def _convert_phone_numbers(text: str) -> str:
    pattern = re.compile(r"(?<!\d)(?:\+84|0)(?:[ .-]?\d){9,10}(?!\d)")

    def replace(match: re.Match[str]) -> str:
        digits = "".join(re.findall(r"\d", match.group(0)))
        if match.group(0).startswith("+84"):
            digits = "0" + digits.removeprefix("84")
        return " ".join(DIGITS[digit] for digit in digits)

    return pattern.sub(replace, text)


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text: str) -> tuple[str, ...]:
    """Split text on sentence-final punctuation, dropping empty chunks."""
    chunks = tuple(chunk.strip() for chunk in SENTENCE_BOUNDARY.split(text))
    filtered = tuple(chunk for chunk in chunks if chunk)
    return filtered or (text.strip(),)


WWW_PREFIX = re.compile(r"^www\.", re.IGNORECASE)


def _speak_dots(raw: str) -> str:
    return " chấm ".join(part for part in raw.split(".") if part)


def _speak_url(match: re.Match[str]) -> str:
    domain = WWW_PREFIX.sub("", match.group(1))
    return f" {_speak_dots(domain)} "


def _speak_emails_and_urls(text: str) -> str:
    """Read emails/URLs aloud instead of deleting them like upstream does."""
    text = re.sub(
        r"(?:https?://|www\.)([^\s/?#]+)[^\s]*",
        _speak_url,
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(
        r"\b([\w.+-]+)@([\w-]+(?:\.[\w-]+)+)",
        lambda match: f"{_speak_dots(match.group(1))} a còng {_speak_dots(match.group(2))}",
        text,
    )


class ValtecTextNormalizer:
    def normalize(self, text: str) -> str:
        normalized = unicodedata.normalize("NFC", text)
        normalized = _speak_emails_and_urls(normalized)
        normalized = normalized.translate(
            str.maketrans({"&": " và ", "@": " a còng ", "#": " thăng ", "_": " "})
        )
        normalized = re.sub(r"[“”„‟]", '"', normalized)
        normalized = re.sub(r"[‘’‚‛]", "'", normalized)
        normalized = re.sub(r"[–—−]", "-", normalized).replace("…", "...")
        normalized = re.sub(r"([!?.])\1+", r"\1", normalized)
        normalized = re.sub(r"[*~`^]", "", normalized)
        normalized = re.sub(
            r"\b(\d{4})\s*-\s*(\d{4})\b",
            lambda match: f"{number_to_words(match.group(1))} đến {number_to_words(match.group(2))}",
            normalized,
        )
        normalized = _convert_dates(normalized)
        normalized = _convert_times(normalized)
        normalized = re.sub(
            r"\b(thứ|lần|bước|phần|chương|tập|số)\s+(\d+)\b(?![.,]\d)",
            lambda match: (
                f"{match.group(1)} {ORDINALS.get(match.group(2), number_to_words(match.group(2)))}"
            ),
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = _convert_currency(normalized)
        normalized = re.sub(
            r"(-?\d+(?:[.,]\d+)?)\s*%",
            lambda match: f"{_spoken_numeric(match.group(1))} phần trăm",
            normalized,
        )
        normalized = re.sub(r"\bwi-?fi\b", "oai phai", normalized, flags=re.IGNORECASE)
        # "vép" is the Vietnamese rendering the Valtec checkpoint can actually
        # pronounce; the English IPA [wɛb] gets swallowed (no final /b/ in VN).
        normalized = re.sub(r"\bweb\b", "vép", normalized, flags=re.IGNORECASE)
        # The checkpoint garbles "linh" in numbers but reads the equivalent
        # "lẻ" wording cleanly; both mean the same connector.
        normalized = re.sub(
            r"\b(trăm|nghìn|ngàn|triệu|tỷ)\s+linh\s+"
            r"(?=(?:một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)\b)",
            r"\1 lẻ ",
            normalized,
        )
        normalized = _convert_phone_numbers(normalized)
        normalized = re.sub(
            r"(?<![\w.,])-?\d+(?:[.,]\d+)+(?![\w.,])",
            lambda match: _spoken_numeric(match.group(0)),
            normalized,
        )
        normalized = re.sub(
            r"(?<!\w)-?\d+(?!\w)",
            lambda match: (
                " ".join(DIGITS[digit] for digit in match.group(0))
                if len(match.group(0)) > 1 and match.group(0).startswith("0")
                else number_to_words(match.group(0))
            ),
            normalized,
        )
        # Compound numbers slur on the Valtec checkpoint; a short rest after
        # each scale word makes them read like well-paced dictation (the same
        # reason digit-by-digit phone numbers already sound clean).
        normalized = re.sub(
            r"\b(trăm|nghìn|ngàn|triệu|tỷ)\s+"
            r"(?=(?:không|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|lẻ|mốt|tư|lăm)\b)",
            r"\1, ",
            normalized,
        )
        return re.sub(r"\s+", " ", normalized).strip()
