import atexit
import contextlib
import importlib.util
import io
import os
import platform
import re
import shutil
import sys
import tempfile
import unicodedata
import warnings
from collections.abc import Callable
from typing import List, Tuple

try:
    import fcntl  # type: ignore
except Exception:
    fcntl = None

vi2IPA: Callable[[str], str] | None = None
VIPHONEME_AVAILABLE = importlib.util.find_spec("viphoneme") is not None
_VIPHONEME_WORKDIR = None
_VINORM_ISOLATED_PARENT = None
_VINORM_RUNTIME_SUPPORTED = None


def _load_vi2ipa() -> Callable[[str], str] | None:
    """Import viphoneme only when it is safe to use.

    Importing viphoneme imports vinorm, whose current wheel emits a Python
    3.12 deprecation warning and may ship a host-incompatible binary. Keep the
    import lazy so unsupported hosts can use the char-based phonemizer without
    importing vinorm at all.
    """
    global VIPHONEME_AVAILABLE, vi2IPA

    if vi2IPA is not None:
        return vi2IPA
    if not VIPHONEME_AVAILABLE:
        return None

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the imp module is deprecated.*",
                category=DeprecationWarning,
            )
            from viphoneme import vi2IPA as imported_vi2IPA
    except Exception:
        VIPHONEME_AVAILABLE = False
        return None

    vi2IPA = imported_vi2IPA
    return vi2IPA


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _vinorm_runtime_supported() -> bool:
    """Return whether the bundled vinorm executable can run on this host.

    The PyPI vinorm wheel currently ships a Linux x86_64 binary named `main`.
    On macOS/Apple Silicon this raises `Exec format error` when viphoneme calls
    vi2IPA. Detect that case before calling viphoneme, then use the local
    character-based phonemizer instead.
    """
    global _VINORM_RUNTIME_SUPPORTED

    if _env_truthy("VIPHONEME_FORCE_VINORM"):
        return True
    if _env_truthy("VIPHONEME_DISABLE_VINORM"):
        return False
    if _VINORM_RUNTIME_SUPPORTED is not None:
        return _VINORM_RUNTIME_SUPPORTED

    supported = True
    spec = importlib.util.find_spec("vinorm")
    if spec is not None and spec.origin is not None:
        main_path = os.path.join(os.path.dirname(spec.origin), "main")
        if os.path.exists(main_path):
            machine = platform.machine().lower()
            supported = sys.platform.startswith("linux") and machine in {"x86_64", "amd64"}

    _VINORM_RUNTIME_SUPPORTED = supported
    return supported


def _get_viphoneme_workdir() -> str:
    global _VIPHONEME_WORKDIR
    if _VIPHONEME_WORKDIR is None:
        _VIPHONEME_WORKDIR = tempfile.mkdtemp(prefix="viphoneme_")
        atexit.register(shutil.rmtree, _VIPHONEME_WORKDIR, ignore_errors=True)
    return _VIPHONEME_WORKDIR


def _ensure_vinorm_isolated() -> None:
    global _VINORM_ISOLATED_PARENT
    if os.environ.get("VIPHONEME_ISOLATE_VINORM", "1") not in {"1", "true", "True", "YES", "yes"}:
        return
    if _VINORM_ISOLATED_PARENT is not None:
        return

    spec = importlib.util.find_spec("vinorm")
    if spec is None or spec.origin is None:
        return

    src_dir = os.path.dirname(spec.origin)
    if not os.path.isfile(os.path.join(src_dir, "__init__.py")):
        return

    parent = tempfile.mkdtemp(prefix="vinorm_")
    dst_dir = os.path.join(parent, "vinorm")
    os.makedirs(dst_dir, exist_ok=True)

    shutil.copy2(os.path.join(src_dir, "__init__.py"), os.path.join(dst_dir, "__init__.py"))

    for name in os.listdir(src_dir):
        if name in {"__init__.py", "__pycache__", "input.txt", "output.txt"}:
            continue
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.exists(dst):
            continue
        try:
            os.symlink(src, dst)
        except Exception:
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            elif os.path.isfile(src):
                shutil.copy2(src, dst)

    if parent not in sys.path:
        sys.path.insert(0, parent)
    if "vinorm" in sys.modules:
        del sys.modules["vinorm"]

    _VINORM_ISOLATED_PARENT = parent
    atexit.register(shutil.rmtree, parent, ignore_errors=True)


@contextlib.contextmanager
def _redirect_fds_to_devnull():
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


@contextlib.contextmanager
def _viphoneme_global_lock():
    lock_path = os.environ.get("VIPHONEME_LOCK_PATH", "/tmp/viphoneme.lock")
    use_lock = os.environ.get("VIPHONEME_USE_LOCK")
    if use_lock is None:
        use_lock = "0" if os.environ.get("VIPHONEME_ISOLATE_VINORM", "1") in {"1", "true", "True", "YES", "yes"} else "1"
    if use_lock not in {"1", "true", "True", "YES", "yes"}:
        yield
        return
    if fcntl is None:
        yield
        return
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# Vietnamese tone diacritics to tone number mapping
TONE_MARKS = {
    '\u0300': 2,  # ̀  huyền (falling)
    '\u0301': 1,  # ́  sắc (rising) 
    '\u0303': 3,  # ̃  ngã (broken)
    '\u0309': 4,  # ̉  hỏi (dipping)
    '\u0323': 5,  # ̣  nặng (heavy/glottalized)
}

# Default tone (no diacritic) = 0 (ngang/level)

# Vietnamese orthography to IPA mapping
VI_TO_IPA = {
    # Trigraphs (check first)
    'ngh': 'ŋ',
    
    # Digraphs
    'ng': 'ŋ',
    'nh': 'ɲ',
    'ch': 'c',      # Vietnamese ch = palatal stop
    'tr': 'ʈ',      # Retroflex
    'th': 'tʰ',     # Aspirated
    'ph': 'f',
    'kh': 'x',      # Voiceless velar fricative
    'gh': 'ɣ',
    'gi': 'z',
    'qu': 'kw',
    
    # Special consonants
    'đ': 'ɗ',       # Implosive d
    
    # Simple consonants
    'b': 'ɓ',       # Implosive b (can also be plain b)
    'c': 'k',
    'd': 'z',       # Northern: z, Southern: j
    'g': 'ɣ',
    'h': 'h',
    'k': 'k',
    'l': 'l',
    'm': 'm',
    'n': 'n',
    'p': 'p',
    'r': 'ʐ',       # Retroflex (varies by dialect)
    's': 's',
    't': 't',
    'v': 'v',
    'x': 's',       # Vietnamese x = s
    
    # Vowels
    'a': 'aː',
    'ă': 'a',       # Short a
    'â': 'ə',       # Schwa
    'e': 'ɛ',
    'ê': 'e',
    'i': 'i',
    'y': 'i',       # Same as i
    'o': 'ɔ',
    'ô': 'o',
    'ơ': 'əː',      # Long schwa
    'u': 'u',
    'ư': 'ɯ',       # Unrounded u
    
    # Diphthongs (handled separately)
}

# Final consonants (codas)
FINAL_CONSONANTS = {
    'c': 'k',
    'ch': 'c',
    'm': 'm',
    'n': 'n',
    'ng': 'ŋ',
    'nh': 'ɲ',
    'p': 'p',
    't': 't',
}

# Punctuation to keep
PUNCTUATION = set(',.!?;:\'"--—…()[]{}')

# Punctuation that creates pauses (SP = short pause)
PAUSE_PUNCTUATION = {',', ';', ':'}
STOP_PUNCTUATION = {'.', '!', '?', '…'}

def extract_tone(char: str) -> Tuple[str, int]:
    """
    Extract tone from a Vietnamese character.
    Returns (base_char, tone_number)
    """
    # Decompose to separate base and combining marks
    decomposed = unicodedata.normalize('NFD', char)
    base = ''
    tone = 0
    
    for c in decomposed:
        if c in TONE_MARKS:
            tone = TONE_MARKS[c]
        elif not unicodedata.combining(c):
            base += c
    
    return base, tone


def syllable_to_ipa(syllable: str) -> Tuple[List[str], int]:
    """
    Convert a Vietnamese syllable to IPA phonemes with tone.
    Returns (phonemes, tone)
    """
    syllable = syllable.lower()
    phonemes = []
    tone = 0
    
    # Extract tone from vowels
    processed = ''
    for char in syllable:
        base, char_tone = extract_tone(char)
        if char_tone > 0:
            tone = char_tone
        processed += base
    
    syllable = processed
    i = 0
    
    while i < len(syllable):
        matched = False
        
        # Try trigraphs
        if i + 2 < len(syllable):
            tri = syllable[i:i+3]
            if tri in VI_TO_IPA:
                phonemes.append(VI_TO_IPA[tri])
                i += 3
                matched = True
        
        # Try digraphs
        if not matched and i + 1 < len(syllable):
            di = syllable[i:i+2]
            if di in VI_TO_IPA:
                phonemes.append(VI_TO_IPA[di])
                i += 2
                matched = True
        
        # Single character
        if not matched:
            char = syllable[i]
            if char in VI_TO_IPA:
                phonemes.append(VI_TO_IPA[char])
            elif char.isalpha():
                phonemes.append(char)  # Keep as-is if not mapped
            i += 1
    
    return phonemes, tone


def text_to_phonemes_viphoneme(text: str) -> Tuple[List[str], List[int], List[int]]:
    """
    Convert text to phonemes using viphoneme library.
    Returns (phones, tones, word2ph)

    viphoneme output format:
    - Syllables separated by space
    - Compound words joined by underscore: hom1_năj1
    - Tone number (1-6) at end of each syllable
    - Punctuation as separate tokens
    """
    # In frozen/PyInstaller mode, use char-based phonemizer directly
    # to avoid vinorm dependency issues
    is_frozen = getattr(sys, 'frozen', False)
    if is_frozen:
        return text_to_phonemes_charbased(text)
    if not _vinorm_runtime_supported():
        return text_to_phonemes_charbased(text)
    converter = _load_vi2ipa()
    if converter is None:
        return text_to_phonemes_charbased(text)
    
    # Normal mode: use full viphoneme with isolation
    try:
        _ensure_vinorm_isolated()
        workdir = _get_viphoneme_workdir()
        with _viphoneme_global_lock():
            cwd = os.getcwd()
            os.chdir(workdir)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with _redirect_fds_to_devnull():
                        ipa_text = converter(text)
            finally:
                os.chdir(cwd)
    except Exception as e:
        print(f"[WARN] Viphoneme failed: {e}")
        return text_to_phonemes_charbased(text)
    
    # Check if viphoneme returned empty or invalid result
    if not ipa_text or ipa_text.strip() in ['', '.', '..', '...']:
        return text_to_phonemes_charbased(text)
    
    phones = []
    tones = []
    word2ph = []
    
    # viphoneme tone mapping: 1=ngang, 2=huyền, 3=ngã, 4=hỏi, 5=sắc, 6=nặng
    # Our internal: 0=ngang, 1=sắc, 2=huyền, 3=ngã, 4=hỏi, 5=nặng
    VIPHONEME_TONE_MAP = {1: 0, 2: 2, 3: 3, 4: 4, 5: 1, 6: 5}
    
    # Characters to skip (combining marks, ties)
    SKIP_CHARS = {'\u0306', '\u0361', '\u032f', '\u0330', '\u0329'}  # breve, tie, etc.
    
    # Split by space
    tokens = ipa_text.strip().split()
    
    for token in tokens:
        # Handle punctuation-only tokens
        if all(c in PUNCTUATION or c == '.' for c in token):
            for c in token:
                if c in PUNCTUATION:
                    phones.append(c)
                    tones.append(0)
                    word2ph.append(1)
            continue
        
        # Split compound words by underscore
        syllables = token.split('_')
        
        for syllable in syllables:
            if not syllable:
                continue
                
            syllable_phones = []
            syllable_tone = 0
            i = 0
            
            while i < len(syllable):
                char = syllable[i]
                
                # Tone number at end
                if char.isdigit():
                    syllable_tone = VIPHONEME_TONE_MAP.get(int(char), 0)
                    i += 1
                    continue
                
                # Skip combining marks (they modify previous char, already handled)
                if unicodedata.combining(char):
                    i += 1
                    continue
                
                # Skip modifier letters like ʷ ʰ (append to previous if exists)
                if char in {'ʷ', 'ʰ', 'ː'}:
                    if syllable_phones:
                        syllable_phones[-1] = syllable_phones[-1] + char
                    i += 1
                    continue
                
                # Skip tie bars and other special marks
                if char in {'\u0361', '\u035c', '\u0361'}:  # tie bars
                    i += 1
                    continue
                
                # Punctuation within syllable
                if char in PUNCTUATION:
                    i += 1
                    continue
                
                # Regular phoneme character
                syllable_phones.append(char)
                i += 1
            
            if syllable_phones:
                phones.extend(syllable_phones)
                tones.extend([syllable_tone] * len(syllable_phones))
                word2ph.append(len(syllable_phones))
    
    return phones, tones, word2ph


def text_to_phonemes_charbased(text: str) -> Tuple[List[str], List[int], List[int]]:
    """
    Convert text to phonemes using character-based mapping.
    Returns (phones, tones, word2ph)
    """
    phones = []
    tones = []
    word2ph = []
    
    words = text.split()
    
    for word in words:
        # Check for punctuation at end
        trailing_punct = []
        while word and word[-1] in PUNCTUATION:
            trailing_punct.insert(0, word[-1])
            word = word[:-1]
        
        # Check for punctuation at start
        leading_punct = []
        while word and word[0] in PUNCTUATION:
            leading_punct.append(word[0])
            word = word[1:]
        
        # Add leading punctuation
        for p in leading_punct:
            phones.append(p)
            tones.append(0)
            word2ph.append(1)
        
        # Process word syllables (Vietnamese words can be multi-syllable)
        if word:
            word_phones, tone = syllable_to_ipa(word)
            if word_phones:
                phones.extend(word_phones)
                tones.extend([tone] * len(word_phones))
                word2ph.append(len(word_phones))
        
        # Add trailing punctuation
        for p in trailing_punct:
            phones.append(p)
            tones.append(0)
            word2ph.append(1)
    
    return phones, tones, word2ph


def text_to_phonemes(text: str, use_viphoneme: bool = True) -> Tuple[List[str], List[int], List[int]]:
    """
    Main function to convert Vietnamese text to phonemes.
    
    Args:
        text: Vietnamese text
        use_viphoneme: Whether to use viphoneme library (if available)
        
    Returns:
        phones: List of IPA phonemes
        tones: List of tone numbers (0-5)
        word2ph: List of phone counts per word
    """
    if use_viphoneme and VIPHONEME_AVAILABLE:
        phones, tones, word2ph = text_to_phonemes_viphoneme(text)
    else:
        phones, tones, word2ph = text_to_phonemes_charbased(text)
    
    # Add boundary tokens
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    
    return phones, tones, word2ph


def get_all_phonemes() -> List[str]:
    """Get list of all possible phonemes for symbol table."""
    phonemes = set()
    
    # From IPA mapping
    for ipa in VI_TO_IPA.values():
        if isinstance(ipa, str):
            phonemes.add(ipa)
            # Also add with length marker
            if len(ipa) == 1:
                phonemes.add(ipa + 'ː')
    
    # Common IPA symbols
    phonemes.update([
        # Consonants
        'b', 'ɓ', 'c', 'd', 'ɗ', 'f', 'g', 'ɣ', 'h', 'j', 'k', 'l', 'm', 'n',
        'ŋ', 'ɲ', 'p', 'r', 'ʐ', 's', 'ʂ', 't', 'tʰ', 'ʈ', 'v', 'w', 'x', 'z',
        # Vowels
        'a', 'aː', 'ə', 'əː', 'ɛ', 'e', 'i', 'ɪ', 'o', 'ɔ', 'u', 'ʊ', 'ɯ', 'ɤ',
        # Special
        '_', ' ',
    ])
    
    # Punctuation
    phonemes.update(PUNCTUATION)
    
    return sorted(list(phonemes))
