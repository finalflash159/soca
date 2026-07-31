"""Acronyms that are read as words rather than letter by letter.

All-caps tokens are letter-spelled by default, which is what a Vietnamese
speaker actually does for "API" or "CPU". A handful are conventionally spoken
as words instead, and no rule distinguishes them -- "JSON" is "jay-son" while
"JSONL" is not. Keys are matched case-sensitively against the all-caps token,
so the English word "post" keeps its own CMU reading.
"""

from __future__ import annotations

ACRONYM_LEXICON: dict[str, str] = {
    "ASCII": "æski",
    "CRUD": "krəd",
    "CUDA": "kudə",
    "DAG": "dæg",
    "FAISS": "feɪs",
    "JSON": "ʤeɪsən",
    "ONNX": "ɑnɪks",
    "POST": "poʊst",
    "RAG": "ræg",
    "REST": "rɛst",
    "TOML": "tɑməl",
    "YAML": "jæməl",
}
