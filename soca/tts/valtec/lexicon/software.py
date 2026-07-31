"""General software, systems and networking vocabulary g2p_en mispredicts.

Two failure families dominate here: lowercase initialisms that the predictor
tries to read as words ("cpu" -> ku, "http" -> tæpti), and long Latinate
derivations where it loses the stem ("reproducibility" -> riprədəsɪtəslaɪt).
"""

from __future__ import annotations

SOFTWARE_LEXICON: dict[str, str] = {
    # Lowercase initialisms: spell the English letter names.
    "api": "eɪpiaɪ",
    "bfs": "biɛfɛs",
    "chmod": "ʧmɑd",
    "cpu": "sipiju",
    "fsync": "ɛfsɪŋk",
    "dfs": "diɛfɛs",
    "dsa": "diɛseɪ",
    "gpu": "ʤipiju",
    "http": "eɪʧtitipi",
    "https": "eɪʧtitipiɛs",
    "kv": "keɪvi",
    "llm": "ɛlɛlɛm",
    "ml": "ɛmɛl",
    "ndcg": "ɛndisiʤi",
    "pid": "piaɪdi",
    "tts": "titiɛs",
    # Letter-plus-word compounds.
    "dtype": "ditaɪp",
    "mmap": "ɛmmæp",
    "mtime": "ɛmtaɪm",
    "sqlite": "ɛskjuɛlaɪt",
    # Derivations where the predictor loses the stem.
    "deduplicate": "didupləkeɪt",
    "deduplication": "didupləkeɪʃən",
    "idempotency": "aɪdɛmpoʊtənsi",
    "idempotent": "aɪdɛmpoʊtənt",
    "initialization": "ɪnɪʃələzeɪʃən",
    "materialization": "mətɪriələzeɪʃən",
    "memoization": "mɛmoʊəzeɪʃən",
    "memoize": "mɛmoʊaɪz",
    "memoized": "mɛmoʊaɪzd",
    "nullability": "nələbɪləti",
    "observability": "əbzərvəbɪləti",
    "reproducibility": "riprədusəbɪləti",
    # Ordinary words the predictor still gets wrong.
    "async": "eɪsɪŋk",
    "atomicity": "ætəmɪsɪti",
    "backspace": "bækspeɪs",
    "bisect": "baɪsɛkt",
    "bitset": "bɪtsɛt",
    "counterexample": "kaʊntərɪgzæmpəl",
    "dedup": "didup",
    "dedupe": "didup",
    "dequeue": "dikju",
    "dijkstra": "daɪkstrə",
    "enqueue": "ɛnkju",
    "eval": "ivæl",
    "filename": "faɪlneɪm",
    "iowait": "aɪoʊweɪt",
    "iterate": "ɪtəreɪt",
    "microbenchmark": "maɪkroʊbɛnʧmɑrk",
    "recompute": "rikəmpjut",
    "redelivery": "ridɪlɪvəri",
    "refcount": "rɛfkaʊnt",
    "regex": "rɛʤɛks",
    "screenshot": "skrinʃɑt",
    "stringify": "strɪŋəfaɪ",
    "subproblem": "səbprɑbləm",
    "syscall": "sɪskɔl",
    "timezone": "taɪmzoʊn",
}
