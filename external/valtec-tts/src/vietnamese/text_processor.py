#!/usr/bin/env python3
"""
Vietnamese Text Processor for TTS
Handles normalization of numbers, dates, times, currencies, etc.
"""

import re
import unicodedata


# Vietnamese number words
DIGITS = {
    '0': 'không', '1': 'một', '2': 'hai', '3': 'ba', '4': 'bốn',
    '5': 'năm', '6': 'sáu', '7': 'bảy', '8': 'tám', '9': 'chín'
}

TEENS = {
    '10': 'mười', '11': 'mười một', '12': 'mười hai', '13': 'mười ba',
    '14': 'mười bốn', '15': 'mười lăm', '16': 'mười sáu', '17': 'mười bảy',
    '18': 'mười tám', '19': 'mười chín'
}

TENS = {
    '2': 'hai mươi', '3': 'ba mươi', '4': 'bốn mươi', '5': 'năm mươi',
    '6': 'sáu mươi', '7': 'bảy mươi', '8': 'tám mươi', '9': 'chín mươi'
}


def number_to_words(num_str):
    """
    Convert a number string to Vietnamese words.
    Handles numbers from 0 to billions.
    """
    # Remove leading zeros but keep at least one digit
    num_str = num_str.lstrip('0') or '0'
    
    # Handle negative numbers
    if num_str.startswith('-'):
        return 'âm ' + number_to_words(num_str[1:])
    
    # Convert to integer for processing
    try:
        num = int(num_str)
    except ValueError:
        return num_str
    
    if num == 0:
        return 'không'
    
    if num < 10:
        return DIGITS[str(num)]
    
    if num < 20:
        return TEENS[str(num)]
    
    if num < 100:
        tens = num // 10
        units = num % 10
        if units == 0:
            return TENS[str(tens)]
        elif units == 1:
            return TENS[str(tens)] + ' mốt'
        elif units == 4:
            return TENS[str(tens)] + ' tư'
        elif units == 5:
            return TENS[str(tens)] + ' lăm'
        else:
            return TENS[str(tens)] + ' ' + DIGITS[str(units)]
    
    if num < 1000:
        hundreds = num // 100
        remainder = num % 100
        result = DIGITS[str(hundreds)] + ' trăm'
        if remainder == 0:
            return result
        elif remainder < 10:
            return result + ' lẻ ' + DIGITS[str(remainder)]
        else:
            return result + ' ' + number_to_words(str(remainder))
    
    if num < 1000000:
        thousands = num // 1000
        remainder = num % 1000
        result = number_to_words(str(thousands)) + ' nghìn'
        if remainder == 0:
            return result
        elif remainder < 10:
            return result + ' lẻ ' + number_to_words(str(remainder))
        elif remainder < 100:
            # Common/Southern style often omits "không trăm" for tens, reading "hai nghìn năm mươi" vs "hai nghìn không trăm năm mươi"
            # However, standard is "không trăm". If user complains, we try removing "không trăm" for smoother reading
            # But let's check if remainder is exactly equivalent to a tens value (e.g. 50, 20).
            # If 2050 -> "hai nghìn năm mươi".
            return result + ' ' + number_to_words(str(remainder)) 
        else:
            return result + ' ' + number_to_words(str(remainder))
    
    if num < 1000000000:
        millions = num // 1000000
        remainder = num % 1000000
        result = number_to_words(str(millions)) + ' triệu'
        if remainder == 0:
            return result
        else:
            return result + ' ' + number_to_words(str(remainder))
    
    if num < 1000000000000:
        billions = num // 1000000000
        remainder = num % 1000000000
        result = number_to_words(str(billions)) + ' tỷ'
        if remainder == 0:
            return result
        else:
            return result + ' ' + number_to_words(str(remainder))
    
    # For very large numbers, read digit by digit
    return ' '.join(DIGITS.get(d, d) for d in num_str)


def convert_decimal(text):
    """Convert decimal numbers: 3.14 -> ba phẩy mười bốn"""
    def replace_decimal(match):
        integer_part = match.group(1)
        decimal_part = match.group(2)
        
        integer_words = number_to_words(integer_part)
        
        # Read decimal part as a number
        decimal_words = number_to_words(decimal_part.lstrip('0') or '0')
        
        return f"{integer_words} phẩy {decimal_words}"
    
    # Match decimal numbers: X.Y where Y is 1-2 digits, followed by space or end
    # Avoid matching large numbers like 100.000 (thousand separator)
    # Match decimal numbers: X.Y or X,Y where Y is 1-2 digits
    # Prioritize this before standalone numbers
    text = re.sub(r'(\d+)[.,](\d{1,2})(?=\s|$|[^\d])', replace_decimal, text)
    return text


def convert_percentage(text):
    """Convert percentages: 50% -> năm mươi phần trăm"""
    def replace_percent(match):
        num = match.group(1)
        return number_to_words(num) + ' phần trăm'
    
    text = re.sub(r'(\d+(?:[.,]\d+)?)\s*%', replace_percent, text)
    return text


def replace_common_terms(text):
    """Replace common terms and abbreviations"""
    # Technology
    text = re.sub(r'\bwi[-]?fi\b', 'oai phai', text, flags=re.IGNORECASE)
    return text


def convert_grouped_numbers(text):
    """
    Handle numbers with separators like 1,000 or 1.000
    Ambiguity is resolved by assuming groups of 3 digits are thousands
    """
    def replace_group(match):
        num_str = match.group(0)
        # Remove separators
        clean_num = re.sub(r'[.,]', '', num_str)
        return number_to_words(clean_num)
    
    # Match numbers like 1.000, 1.000.000 or 1,000, 1,000,000
    # Must match at least one separator and exactly 3 digits after it
    # We use negative lookahead/lookbehind to avoid matching parts of decimals/dates if possible
    text = re.sub(r'(?<![\d.,])\d{1,3}(?:[.,]\d{3})+(?![\d.,])', replace_group, text)
    return text



def convert_currency(text):
    """Convert currency amounts"""
    # Vietnamese Dong - be specific to avoid matching "đ" in other words like "độ"
    def replace_vnd(match):
        num = match.group(1).replace('.', '').replace(',', '')
        return number_to_words(num) + ' đồng'
    
    # Only match currency patterns: number followed by currency symbol at word boundary
    text = re.sub(r'(\d+(?:[.,]\d+)*)\s*(?:đồng|VND|vnđ)\b', replace_vnd, text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+(?:[.,]\d+)*)đ(?![a-zà-ỹ])', replace_vnd, text, flags=re.IGNORECASE)
    
    # USD
    def replace_usd(match):
        num = match.group(1).replace('.', '').replace(',', '')
        return number_to_words(num) + ' đô la'
    
    text = re.sub(r'\$\s*(\d+(?:[.,]\d+)*)', replace_usd, text)
    text = re.sub(r'(\d+(?:[.,]\d+)*)\s*(?:USD|\$)', replace_usd, text, flags=re.IGNORECASE)
    
    return text


def convert_time(text):
    """Convert time expressions: 2 giờ 20 phút -> hai giờ hai mươi phút"""
    def replace_time(match):
        hour = match.group(1)
        minute = match.group(2) if match.group(2) else None
        second = match.group(3) if len(match.groups()) > 2 and match.group(3) else None
        
        result = number_to_words(hour) + ' giờ'
        if minute:
            result += ' ' + number_to_words(minute) + ' phút'
        if second:
            result += ' ' + number_to_words(second) + ' giây'
        return result
    
    # HH:MM:SS or HH:MM
    text = re.sub(r'(\d{1,2}):(\d{2})(?::(\d{2}))?', replace_time, text)
    
    # X giờ Y phút
    def replace_time_vn(match):
        hour = match.group(1)
        minute = match.group(2)
        return number_to_words(hour) + ' giờ ' + number_to_words(minute) + ' phút'
    
    text = re.sub(r'(\d+)\s*giờ\s*(\d+)\s*phút', replace_time_vn, text)
    
    # X giờ (without minute)
    def replace_hour(match):
        hour = match.group(1)
        return number_to_words(hour) + ' giờ'
    
    text = re.sub(r'(\d+)\s*giờ(?!\s*\d)', replace_hour, text)
    
    return text


def convert_date(text):
    """Convert date expressions"""
    # DD/MM/YYYY or DD-MM-YYYY
    def replace_date_full(match):
        day = match.group(1)
        month = match.group(2)
        year = match.group(3)
        return f"ngày {number_to_words(day)} tháng {number_to_words(month)} năm {number_to_words(year)}"
    
    # First, replace "Sinh ngày DD/MM/YYYY" pattern to avoid double "ngày"
    text = re.sub(r'(Sinh|sinh)\s+ngày\s+(\d{1,2})[/-](\d{1,2})[/-](\d{4})', 
                  lambda m: f"{m.group(1)} ngày {number_to_words(m.group(2))} tháng {number_to_words(m.group(3))} năm {number_to_words(m.group(4))}", text)
    
    text = re.sub(r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})', replace_date_full, text)
    
    # X tháng Y
    def replace_month_day(match):
        day = match.group(1)
        month = match.group(2)
        return f"ngày {number_to_words(day)} tháng {number_to_words(month)}"
    
    text = re.sub(r'(\d+)\s*tháng\s*(\d+)', replace_month_day, text)
    
    # tháng X (month only)
    def replace_month(match):
        month = match.group(1)
        return 'tháng ' + number_to_words(month)
    
    text = re.sub(r'tháng\s*(\d+)', replace_month, text)
    
    # ngày X
    def replace_day(match):
        day = match.group(1)
        return 'ngày ' + number_to_words(day)
    
    text = re.sub(r'ngày\s*(\d+)', replace_day, text)
    
    return text


def convert_year_range(text):
    """Convert year ranges: 1873-1907 -> một nghìn tám trăm bảy mươi ba đến một nghìn chín trăm lẻ bảy"""
    def replace_year_range(match):
        year1 = match.group(1)
        year2 = match.group(2)
        return number_to_words(year1) + ' đến ' + number_to_words(year2)
    
    text = re.sub(r'(\d{4})\s*[-–—]\s*(\d{4})', replace_year_range, text)
    return text


def convert_ordinal(text):
    """Convert ordinals: thứ 2 -> thứ hai"""
    ordinal_map = {
        '1': 'nhất', '2': 'hai', '3': 'ba', '4': 'tư', '5': 'năm',
        '6': 'sáu', '7': 'bảy', '8': 'tám', '9': 'chín', '10': 'mười'
    }
    
    def replace_ordinal(match):
        prefix = match.group(1)
        num = match.group(2)
        if num in ordinal_map:
            return prefix + ' ' + ordinal_map[num]
        return prefix + ' ' + number_to_words(num)
    
    # thứ X, lần X, bước X, phần X
    text = re.sub(r'(thứ|lần|bước|phần|chương|tập|số)\s*(\d+)', replace_ordinal, text, flags=re.IGNORECASE)
    
    return text


def convert_standalone_numbers(text):
    """Convert remaining standalone numbers to words"""
    def replace_num(match):
        num = match.group(0)
        # Skip if it's part of a word or already processed
        return number_to_words(num)
    
    # Match numbers not followed/preceded by letters
    text = re.sub(r'\b\d+\b', replace_num, text)
    return text


def convert_phone_number(text):
    """Read phone numbers digit by digit"""
    def replace_phone(match):
        phone = match.group(0)
        digits = re.findall(r'\d', phone)
        return ' '.join(DIGITS.get(d, d) for d in digits)
    
    # Vietnamese phone patterns
    text = re.sub(r'0\d{9,10}', replace_phone, text)
    text = re.sub(r'\+84\d{9,10}', replace_phone, text)
    
    return text


def normalize_unicode(text):
    """Normalize Unicode to NFC form"""
    return unicodedata.normalize('NFC', text)


def clean_whitespace(text):
    """Clean up extra whitespace"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_special_chars(text):
    """Remove or replace special characters that can't be spoken"""
    # Keep Vietnamese diacritics and common punctuation
    # Remove emojis and special symbols
    
    # Replace common symbols with words
    text = text.replace('&', ' và ')
    text = text.replace('@', ' a còng ')
    text = text.replace('#', ' thăng ')
    text = text.replace('*', '')
    text = text.replace('_', ' ')
    text = text.replace('~', '')
    text = text.replace('`', '')
    text = text.replace('^', '')
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    return text


def normalize_punctuation(text):
    """Normalize punctuation marks"""
    # Normalize quotes
    text = re.sub(r'[""„‟]', '"', text)
    text = re.sub(r"[''‚‛]", "'", text)
    
    # Normalize dashes
    text = re.sub(r'[–—−]', '-', text)
    
    # Normalize ellipsis
    text = re.sub(r'\.{3,}', '...', text)
    text = text.replace('…', '...')
    
    # Remove multiple punctuation
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    
    return text


def process_vietnamese_text(text):
    """
    Main function to process Vietnamese text for TTS.
    Applies all normalization steps in the correct order.
    
    Args:
        text: Raw Vietnamese text
        
    Returns:
        Normalized text suitable for TTS
    """
    # Step 1: Normalize Unicode
    text = normalize_unicode(text)
    
    # Step 2: Remove special characters
    text = remove_special_chars(text)
    
    # Step 3: Normalize punctuation
    text = normalize_punctuation(text)
    
    # Step 4: Convert year ranges (before other number conversions)
    text = convert_year_range(text)
    
    # Step 5: Convert dates
    text = convert_date(text)
    
    # Step 6: Convert times
    text = convert_time(text)
    
    # Step 7: Convert ordinals
    text = convert_ordinal(text)
    
    # Step 8: Convert currency
    text = convert_currency(text)
    
    # Step 9: Convert percentages
    text = convert_percentage(text)
    
    # Step 10: Convert common terms (wifi)
    text = replace_common_terms(text)
    
    # Step 11: Convert phone numbers
    text = convert_phone_number(text)
    
    # Step 12: Convert grouped numbers (1,000 -> 1000)
    # Must be before decimals and standalone
    text = convert_grouped_numbers(text)
    
    # Step 13: Convert decimals (before standalone numbers)
    # Now handles both . and , for decimals
    text = convert_decimal(text)
    
    # Step 14: Convert remaining standalone numbers
    text = convert_standalone_numbers(text)
    
    # Step 15: Clean whitespace
    text = clean_whitespace(text)
    
    return text


if __name__ == "__main__":
    # Test cases
    test_cases = [
        "Lúc khoảng 2 giờ 20 phút sáng ngày thứ Bảy hay 8 tháng 11",
        "Alfred Jarry 1873-1907 hợp những nhà văn",
        "ông Derringer 44 ly, dí sát đầu tổng thống",
        "Giá sản phẩm là 100.000đ",
        "Tỷ lệ thành công đạt 85%",
        "Họp lúc 14:30",
        "Sinh ngày 15/08/1990",
        "Chương 3: Hành trình mới",
        "Số điện thoại: 0912345678",
        "Nhiệt độ 25.5 độ C",
        "Công ty XYZ có 1500 nhân viên",
    ]
    
    print("=" * 60)
    print("Vietnamese Text Processor Test")
    print("=" * 60)
    
    for text in test_cases:
        processed = process_vietnamese_text(text)
        print(f"\nOriginal: {text}")
        print(f"Processed: {processed}")
