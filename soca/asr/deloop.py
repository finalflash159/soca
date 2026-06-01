"""De-looping for Whisper output: collapse consecutively repeated fragments.

Whisper hallucinations commonly manifest as repeated token windows
(e.g. "thấy thế nào thấy thế nào thấy thế nào"). De-looping removes
these without touching legitimate short repeats.

Reference: Barański et al. (ICASSP 2025), Sec. II-A.
"""

from __future__ import annotations


def remove_consecutive_repeats(text: str, max_token_len: int = 6) -> tuple[str, bool]:
    """Collapse consecutive repeated token windows. Keep one copy.

    Looks for the longest repeating window starting at each position
    (window length from `max_token_len` down to 2). When a window matches
    the next `window_len` tokens, drop the duplicates.

    Args:
        text: Decoder output text (whitespace-tokenized).
        max_token_len: Longest window length to consider. Keep small to
            avoid false positives on long natural phrases.

    Returns:
        (cleaned_text, was_looping). `was_looping=True` is a useful signal
        to surface in diagnostics — even after cleanup, the fact that the
        decoder looped at all means quality may have been degraded.

    Examples:
        >>> remove_consecutive_repeats("xin chào xin chào")
        ('xin chào', True)
        >>> remove_consecutive_repeats("đặt báo thức lúc bảy giờ")
        ('đặt báo thức lúc bảy giờ', False)
    """
    tokens = text.split()
    if len(tokens) < 4:
        return text, False

    out_tokens: list[str] = []
    i = 0
    looping_detected = False

    while i < len(tokens):
        found_repeat = False
        max_window = min(max_token_len, (len(tokens) - i) // 2)
        for window_len in range(max_window, 1, -1):
            window = tokens[i : i + window_len]
            next_window = tokens[i + window_len : i + 2 * window_len]
            if window == next_window:
                out_tokens.extend(window)
                looping_detected = True
                j = i + window_len
                while tokens[j : j + window_len] == window:
                    j += window_len
                i = j
                found_repeat = True
                break

        if not found_repeat:
            out_tokens.append(tokens[i])
            i += 1

    return " ".join(out_tokens), looping_detected


def has_excessive_repetition(text: str, threshold: float = 0.45) -> bool:
    """Crude unigram-repetition check used as a fallback before/after deloop.

    `remove_consecutive_repeats` only handles strict consecutive windows.
    This catches "scattered" repetition (same word reused 6 times out of 10)
    that escapes the structural deloop.
    """
    tokens = text.lower().split()
    if len(tokens) < 4:
        return False
    return (1.0 - len(set(tokens)) / len(tokens)) > threshold
