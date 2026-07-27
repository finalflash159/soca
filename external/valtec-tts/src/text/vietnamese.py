import re
import unicodedata
from transformers import AutoTokenizer
from . import punctuation, symbols

# Vietnamese BERT model
model_id = 'vinai/phobert-base-v2'
tokenizer = None

def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer

# Vietnamese IPA phoneme set based on VieNeu-TTS-140h dataset
# These are extracted from the phonemized_text field in the dataset
VI_IPA_CONSONANTS = [
    'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'w', 'x', 'z',
    'ŋ',  # ng
    'ɲ',  # nh
    'ʈ',  # tr
    'ɖ',  # đ
    'tʰ', # th
    'kʰ', # kh
    'ʂ',  # s (southern)
    'ɣ',  # g (southern)
    'χ',  # x (some dialects)
]

VI_IPA_VOWELS = [
    'a', 'ă', 'â', 'e', 'ê', 'i', 'o', 'ô', 'ơ', 'u', 'ư', 'y',
    'ə',  # ơ
    'ɛ',  # e
    'ɔ',  # o
    'ɯ',  # ư
    'ɤ',  # ơ variant
    'ɐ',  # a short
    'ʊ',  # u short
    'ɪ',  # i short
    'ʌ',  # â
    'æ',  # a variant
]

# Vietnamese tone markers (numbers 1-6 or ˈ ˌ for stress)
VI_TONE_MARKERS = ['1', '2', '3', '4', '5', '6', 'ˈ', 'ˌ', 'ː']

# Combined IPA symbols used in VieNeu-TTS dataset
VI_IPA_SYMBOLS = [
    # Consonants
    'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'w', 'x', 'z',
    'ŋ', 'ɲ', 'ʈ', 'ɖ', 'ʂ', 'ɣ', 'χ', 'ʔ',
    # Vowels
    'a', 'ă', 'e', 'i', 'o', 'u', 'y',
    'ə', 'ɛ', 'ɔ', 'ɯ', 'ɤ', 'ɐ', 'ʊ', 'ɪ', 'ʌ', 'æ', 'ɑ',
    # Special markers
    'ˈ', 'ˌ', 'ː',
    # Tone numbers
    '1', '2', '3', '4', '5', '6',
]

def normalize_vietnamese_text(text):
    """Normalize Vietnamese text."""
    # Normalize unicode
    text = unicodedata.normalize('NFC', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Convert numbers to words (basic)
    text = convert_numbers_to_vietnamese(text)
    
    return text

def convert_numbers_to_vietnamese(text):
    """Convert numbers to Vietnamese words (basic implementation)."""
    num_map = {
        '0': 'không', '1': 'một', '2': 'hai', '3': 'ba', '4': 'bốn',
        '5': 'năm', '6': 'sáu', '7': 'bảy', '8': 'tám', '9': 'chín',
        '10': 'mười', '100': 'trăm', '1000': 'nghìn'
    }
    
    # Simple replacement for single digits in context
    def replace_num(match):
        num = match.group(0)
        if num in num_map:
            return num_map[num]
        return num
    
    # Only replace standalone numbers
    text = re.sub(r'\b\d\b', replace_num, text)
    return text

def text_normalize(text):
    """Normalize text for Vietnamese TTS."""
    text = normalize_vietnamese_text(text)
    return text

def parse_ipa_phonemes(phonemized_text):
    """
    Parse IPA phonemized text from VieNeu-TTS dataset.
    Example: "ŋˈyə2j ŋˈyə2j bˈan xwˈan vˈe2"
    Returns: phones, tones, word2ph
    """
    phones = []
    tones = []
    word2ph = []
    
    # Split by space to get words
    words = phonemized_text.strip().split()
    
    for word in words:
        word_phones = []
        word_tones = []
        
        # Parse each character/symbol in the word
        i = 0
        current_tone = 0  # Default tone (neutral/tone 1)
        
        while i < len(word):
            char = word[i]
            
            # Check for tone numbers (1-6)
            if char.isdigit():
                current_tone = int(char)
                i += 1
                continue
            
            # Check for stress markers
            if char in ['ˈ', 'ˌ']:
                # Primary or secondary stress - could be used as tone variant
                i += 1
                continue
            
            # Check for length marker
            if char == 'ː':
                # Long vowel marker - append to previous phone if exists
                if word_phones:
                    word_phones[-1] = word_phones[-1] + 'ː'
                i += 1
                continue
            
            # Check for punctuation
            if char in punctuation:
                if word_phones:
                    phones.extend(word_phones)
                    tones.extend([current_tone] * len(word_phones))
                    word2ph.append(len(word_phones))
                    word_phones = []
                    word_tones = []
                phones.append(char)
                tones.append(0)
                word2ph.append(1)
                i += 1
                continue
            
            # Regular phoneme
            word_phones.append(char)
            i += 1
        
        # Apply collected tone to all phones in this word
        if word_phones:
            phones.extend(word_phones)
            tones.extend([current_tone] * len(word_phones))
            word2ph.append(len(word_phones))
    
    return phones, tones, word2ph

def g2p_ipa(text):
    """
    Convert text to phonemes using external IPA converter.
    This is a fallback for when phonemized_text is not available.
    For training, we use the pre-phonemized text from the dataset.
    """
    try:
        from viphoneme import vi2ipa
        phonemized = vi2ipa(text)
        phones, tones, word2ph = parse_ipa_phonemes(phonemized)
    except ImportError:
        # Fallback: use character-based representation
        phones, tones, word2ph = g2p_char_based(text)
    
    # Add start and end tokens
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    
    return phones, tones, word2ph

def g2p_char_based(text):
    """
    Character-based G2P with Vietnamese to IPA mapping.
    """
    phones = []
    tones = []
    word2ph = []
    
    # Vietnamese tone marks to tone number mapping
    tone_marks = {
        '\u0300': 2,  # à - huyền
        '\u0301': 1,  # á - sắc  
        '\u0303': 3,  # ã - ngã
        '\u0309': 4,  # ả - hỏi
        '\u0323': 5,  # ạ - nặng
    }
    
    # Vietnamese character to IPA mapping (COMPREHENSIVE - matching training data)
    # Multi-char outputs are split into lists to avoid KeyError for missing multi-char symbols
    vi_to_ipa = {
        # Multi-char consonants (check these first - ORDER MATTERS)
        'ngh': 'ŋ',
        'ng': 'ŋ',
        'nh': 'ɲ',
        'ch': ['t', 'ʃ'],  # Vietnamese ch = IPA t + ʃ (separated in training data)
        'tr': 'ʈ',   # retroflex
        'th': ['t', 'h'],   # aspirated th
        'ph': 'f',
        'kh': 'x',   # Vietnamese 'kh' = IPA 'x' (matches training data)
        'gh': 'ɣ',
        'gi': 'z',
        'qu': 'kw',   # qu -> kw (single symbol in training data)
        # Special Vietnamese consonants
        'đ': 'ɗ',    # implosive d
        # Basic consonants that need IPA mapping
        'x': 's',    # Vietnamese 'x' = IPA 's'
        'c': 'k',    # Vietnamese 'c' = IPA 'k'
        'd': 'z',    # Vietnamese 'd' (northern) = 'z'
        'r': 'ɹ',    # Vietnamese 'r' = IPA 'ɹ' (matches training data)
        's': 's',
        'b': 'b',
        'g': 'ɣ',
        'h': 'h',
        'k': 'k',
        'l': 'l',
        'm': 'm',
        'n': 'n',
        'p': 'p',
        't': 't',
        'v': 'v',
        'f': 'f',
        'j': 'j',
        'w': 'w',
        'y': 'j',    # Vietnamese 'y' = IPA 'j' (matches training data)
        # Vowels - MUST match training data phonemes exactly!
        'a': 'aː',   # Long 'a' (matches training: aː)
        'ă': 'a',    # Short 'a' 
        'â': 'ə',    # schwa
        'e': 'ɛ',    # open-mid (matches training: ɛ)
        'ê': 'e',    # close-mid
        'i': 'i',
        'o': 'ɔ',    # open-mid back (matches training: ɔ)
        'ô': 'o',    # close-mid back
        'ơ': 'əː',   # long schwa
        'u': 'u',
        'ư': 'ɯ',    # close back unrounded
    }
    
    words = text.split()
    for word in words:
        # Decompose to separate base char and tone mark
        decomposed = unicodedata.normalize('NFD', word)
        word_phones = []
        current_tone = 0
        
        i = 0
        chars = list(decomposed)
        while i < len(chars):
            char = chars[i]
            
            if char in tone_marks:
                current_tone = tone_marks[char]
                i += 1
                continue
            
            if char in punctuation:
                if word_phones:
                    phones.extend(word_phones)
                    tones.extend([current_tone] * len(word_phones))
                    word2ph.append(len(word_phones))
                    word_phones = []
                phones.append(char)
                tones.append(0)
                word2ph.append(1)
                current_tone = 0
                i += 1
                continue
            
            if unicodedata.combining(char):
                i += 1
                continue
            
            # Check for multi-char sequences (digraphs/trigraphs)
            lower_char = char.lower()
            matched = False
            
            # Try trigraphs first
            if i + 2 < len(chars):
                trigraph = (lower_char + chars[i+1].lower() + chars[i+2].lower())
                if trigraph in vi_to_ipa:
                    result = vi_to_ipa[trigraph]
                    if isinstance(result, list):
                        word_phones.extend(result)
                    else:
                        word_phones.append(result)
                    i += 3
                    matched = True
            
            # Try digraphs
            if not matched and i + 1 < len(chars):
                digraph = lower_char + chars[i+1].lower()
                if digraph in vi_to_ipa:
                    result = vi_to_ipa[digraph]
                    if isinstance(result, list):
                        word_phones.extend(result)
                    else:
                        word_phones.append(result)
                    i += 2
                    matched = True
            
            # Single char
            if not matched:
                if lower_char in vi_to_ipa:
                    result = vi_to_ipa[lower_char]
                    if isinstance(result, list):
                        word_phones.extend(result)
                    else:
                        word_phones.append(result)
                else:
                    word_phones.append(lower_char)
                i += 1
        
        if word_phones:
            phones.extend(word_phones)
            tones.extend([current_tone] * len(word_phones))
            word2ph.append(len(word_phones))
    
    # Add boundary tokens
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    
    return phones, tones, word2ph

def g2p(text):
    """
    Main G2P function for Vietnamese.
    Uses character-to-IPA mapping with BERT alignment.
    """
    tok = get_tokenizer()
    norm_text = text_normalize(text)
    
    # Tokenize for BERT alignment
    tokenized = tok.tokenize(norm_text)
    
    # Use character-based G2P with IPA mapping
    phones, tones, word2ph = g2p_char_based(norm_text)
    
    # Ensure word2ph aligns with tokenized output
    # PhoBERT uses subword tokenization, so we need to distribute phones
    if len(word2ph) != len(tokenized) + 2:  # +2 for start/end tokens
        # Redistribute word2ph to match tokenized length
        total_phones = sum(word2ph)
        new_word2ph = distribute_phones(total_phones, len(tokenized))
        word2ph = [1] + new_word2ph + [1]
    
    return phones, tones, word2ph

def g2p_with_phonemes(text, phonemized_text):
    """
    G2P using pre-phonemized text from dataset.
    This is the recommended method for training.
    """
    tok = get_tokenizer()
    
    # Parse IPA phonemes
    phones, tones, word2ph = parse_ipa_phonemes(phonemized_text)
    
    # Add boundary tokens
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    
    # Get tokenized text for BERT alignment
    tokenized = tok.tokenize(text)
    
    # Distribute word2ph to match tokenized output + boundaries
    if word2ph:
        total_phones = sum(word2ph)
        new_word2ph = distribute_phones(total_phones, len(tokenized))
        word2ph = [1] + new_word2ph + [1]
    else:
        word2ph = [1] + [1] * len(tokenized) + [1]
    
    return phones, tones, word2ph

def distribute_phones(n_phone, n_word):
    """Distribute phones across words as evenly as possible."""
    if n_word == 0:
        return []
    phones_per_word = [n_phone // n_word] * n_word
    remainder = n_phone % n_word
    for i in range(remainder):
        phones_per_word[i] += 1
    return phones_per_word

def get_bert_feature(text, word2ph, device='cuda'):
    """Get BERT features for Vietnamese text."""
    from . import vietnamese_bert
    return vietnamese_bert.get_bert_feature(text, word2ph, device=device, model_id=model_id)


if __name__ == "__main__":
    # Test
    test_text = "Xin chào, tôi là một trợ lý AI."
    test_phonemes = "sˈin tʂˈaːw, tˈoj lˈaː2 mˈo6t tʂˈɤ4 lˈi4 ˌaːˈi."
    
    print("Test text:", test_text)
    print("Normalized:", text_normalize(test_text))
    
    # Test with phonemes
    phones, tones, word2ph = g2p_with_phonemes(test_text, test_phonemes)
    print("Phones:", phones)
    print("Tones:", tones)
    print("Word2Ph:", word2ph)
    
    # Test without phonemes
    phones2, tones2, word2ph2 = g2p(test_text)
    print("\nChar-based phones:", phones2)
    print("Char-based tones:", tones2)
