"""
Vietnamese language support package
"""

from .phonemizer import VIPHONEME_AVAILABLE, get_all_phonemes, text_to_phonemes
from .text_processor import process_vietnamese_text
