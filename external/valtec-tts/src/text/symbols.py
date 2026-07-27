# punctuation = ["!", "?", "…", ",", ".", "'", "-"]
punctuation = ["!", "?", "…", ",", ".", "'", "-", "¿", "¡"]
pu_symbols = punctuation + ["SP", "UNK"]
pad = "_"

# chinese
zh_symbols = [
    "E",
    "En",
    "a",
    "ai",
    "an",
    "ang",
    "ao",
    "b",
    "c",
    "ch",
    "d",
    "e",
    "ei",
    "en",
    "eng",
    "er",
    "f",
    "g",
    "h",
    "i",
    "i0",
    "ia",
    "ian",
    "iang",
    "iao",
    "ie",
    "in",
    "ing",
    "iong",
    "ir",
    "iu",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "ong",
    "ou",
    "p",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "u",
    "ua",
    "uai",
    "uan",
    "uang",
    "ui",
    "un",
    "uo",
    "v",
    "van",
    "ve",
    "vn",
    "w",
    "x",
    "y",
    "z",
    "zh",
    "AA",
    "EE",
    "OO",
]
num_zh_tones = 6

# japanese
ja_symbols = [
    "N",
    "a",
    "a:",
    "b",
    "by",
    "ch",
    "d",
    "dy",
    "e",
    "e:",
    "f",
    "g",
    "gy",
    "h",
    "hy",
    "i",
    "i:",
    "j",
    "k",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "o:",
    "p",
    "py",
    "q",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "ty",
    "u",
    "u:",
    "w",
    "y",
    "z",
    "zy",
]
num_ja_tones = 1

# English
en_symbols = [
    "aa",
    "ae",
    "ah",
    "ao",
    "aw",
    "ay",
    "b",
    "ch",
    "d",
    "dh",
    "eh",
    "er",
    "ey",
    "f",
    "g",
    "hh",
    "ih",
    "iy",
    "jh",
    "k",
    "l",
    "m",
    "n",
    "ng",
    "ow",
    "oy",
    "p",
    "r",
    "s",
    "sh",
    "t",
    "th",
    "uh",
    "uw",
    "V",
    "w",
    "y",
    "z",
    "zh",
]
num_en_tones = 4

# Korean
kr_symbols = ['ᄌ', 'ᅥ', 'ᆫ', 'ᅦ', 'ᄋ', 'ᅵ', 'ᄅ', 'ᅴ', 'ᄀ', 'ᅡ', 'ᄎ', 'ᅪ', 'ᄑ', 'ᅩ', 'ᄐ', 'ᄃ', 'ᅢ', 'ᅮ', 'ᆼ', 'ᅳ', 'ᄒ', 'ᄆ', 'ᆯ', 'ᆷ', 'ᄂ', 'ᄇ', 'ᄉ', 'ᆮ', 'ᄁ', 'ᅬ', 'ᅣ', 'ᄄ', 'ᆨ', 'ᄍ', 'ᅧ', 'ᄏ', 'ᆸ', 'ᅭ', '(', 'ᄊ', ')', 'ᅲ', 'ᅨ', 'ᄈ', 'ᅱ', 'ᅯ', 'ᅫ', 'ᅰ', 'ᅤ', '~', '\\', '[', ']', '/', '^', ':', 'ㄸ', '*']
num_kr_tones = 1

# Spanish
es_symbols = [
        "N",
        "Q",
        "a",
        "b",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "ɑ",
        "æ",
        "ʃ",
        "ʑ",
        "ç",
        "ɯ",
        "ɪ",
        "ɔ",
        "ɛ",
        "ɹ",
        "ð",
        "ə",
        "ɫ",
        "ɥ",
        "ɸ",
        "ʊ",
        "ɾ",
        "ʒ",
        "θ",
        "β",
        "ŋ",
        "ɦ",
        "ɡ",
        "r",
        "ɲ",
        "ʝ",
        "ɣ",
        "ʎ",
        "ˈ",
        "ˌ",
        "ː"
    ]
num_es_tones = 1

# French 
fr_symbols = [
    "\u0303",
    "œ",
    "ø",
    "ʁ",
    "ɒ",
    "ʌ",
    "ɜ",
    "ɐ"
]
num_fr_tones = 1

# German 
de_symbols = [
    "ʏ",
    "̩"
  ]
num_de_tones = 1

# Russian 
ru_symbols = [
    "ɭ",
    "ʲ",
    "ɕ",
    "\"",
    "ɵ",
    "^",
    "ɬ"
]
num_ru_tones = 1

# Vietnamese (IPA-based, compatible with VieNeu-TTS-140h dataset)
vi_symbols = [
    # Consonants (simple)
    "ʈ",   # tr
    "ɖ",   # đ
    "ɗ",   # implosive d (đ variant)
    "ɓ",   # implosive b
    "ʰ",   # aspiration marker
    "ă",   # short a (Vietnamese)
    "ʷ",   # labialization marker
    "̆",    # breve diacritic
    "͡",    # tie bar (for affricates)
    "ʤ",   # voiced postalveolar affricate
    "ʧ",   # voiceless postalveolar affricate
    # Foreign/special characters found in dataset
    "т",   # Cyrillic т
    "輪",  # Chinese character
    "и",   # Cyrillic и
    "л",   # Cyrillic л
    "р",   # Cyrillic р
    "µ",   # micro sign
    "ʂ",   # s (retroflex)
    "ʐ",   # r (retroflex)
    "ʔ",   # glottal stop
    "ɣ",   # g (southern)
    # Multi-char consonants (from vietnamese.py g2p)
    "tʰ",  # th
    "kʰ",  # kh
    "kw",  # qu -> kw
    "tʃ",  # ch
    "ɹ",   # r IPA
    # Vowels specific to Vietnamese
    "ɤ",   # ơ
    "ɐ",   # a short
    "ɑ",   # a back
    "ɨ",   # ư variant
    "ʉ",   # u variant
    "ɜ",   # open-mid central
    # Long vowels (from VieNeu-TTS dataset)
    "əː",  # schwa long
    "aː",  # a long  
    "ɜː",  # open-mid central long
    "ɑː",  # open back long
    "ɔː",  # open-mid back long
    "iː",  # close front long
    "uː",  # close back long
    "eː",  # close-mid front long
    "oː",  # close-mid back long
    # Diphthongs and special combinations
    "iə",  # ia/iê
    "ɨə",  # ưa/ươ
    "uə",  # ua/uô
    # Additional IPA markers
    "ˑ",   # half-long
    "̪",    # dental diacritic
    # Tone-related (though tones are handled separately)
    "˥",   # tone 1 marker
    "˩",   # tone marker
    "˧",   # tone marker
    "˨",   # tone marker
    "˦",   # tone marker
    # Numbers (found in phonemized dataset)
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    # Special characters from dataset
    "$", "%", "&", "«", "»", "–", "ı",
    # viphoneme specific symbols
    "wʷ",  # labialized w
    "#",   # unknown/fallback marker
    "ô",   # Vietnamese ô (fallback)
    "ʃ",   # voiceless postalveolar fricative
    "ʒ",   # voiced postalveolar fricative
    "θ",   # voiceless dental fricative
    "ð",   # voiced dental fricative
    "æ",   # near-open front unrounded
    "ɪ",   # near-close front unrounded
    "ʊ",   # near-close back rounded 
    # Vietnamese fallback characters (when viphoneme fails to parse)
    "ẩ", "ò", "à", "á", "ủ", "ờ", "ộ", "ả", "ó", "é", "ê",
    "ồ", "ấ", "ú", "ế", "ớ", "ì", "ọ", "ố", "ư", "ữ",
]
num_vi_tones = 8  # 6 tones + 1 neutral + 1 extra for data compatibility

# combine all symbols
normal_symbols = sorted(set(zh_symbols + ja_symbols + en_symbols + kr_symbols + es_symbols + fr_symbols + de_symbols + ru_symbols + vi_symbols))
symbols = [pad] + normal_symbols + pu_symbols
sil_phonemes_ids = [symbols.index(i) for i in pu_symbols]

# combine all tones
num_tones = num_zh_tones + num_ja_tones + num_en_tones + num_kr_tones + num_es_tones + num_fr_tones + num_de_tones + num_ru_tones + num_vi_tones

# language maps
language_id_map = {"ZH": 0, "JP": 1, "EN": 2, "ZH_MIX_EN": 3, 'KR': 4, 'ES': 5, 'SP': 5, 'FR': 6, 'VI': 7}
num_languages = len(language_id_map.keys())

language_tone_start_map = {
    "ZH": 0,
    "ZH_MIX_EN": 0,
    "JP": num_zh_tones,
    "EN": num_zh_tones + num_ja_tones,
    'KR': num_zh_tones + num_ja_tones + num_en_tones,
    "ES": num_zh_tones + num_ja_tones + num_en_tones + num_kr_tones,
    "SP": num_zh_tones + num_ja_tones + num_en_tones + num_kr_tones,
    "FR": num_zh_tones + num_ja_tones + num_en_tones + num_kr_tones + num_es_tones,
    "VI": num_zh_tones + num_ja_tones + num_en_tones + num_kr_tones + num_es_tones + num_fr_tones + num_de_tones + num_ru_tones,
}

if __name__ == "__main__":
    a = set(zh_symbols)
    b = set(en_symbols)
    print(sorted(a & b))
