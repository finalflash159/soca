"""Curated pronunciations for names statistical G2P cannot derive.

g2p_en predicts from English spelling rules learned over CMUdict. Brand names
and technical coinages break those rules on purpose: "github" is git+hub, not a
word containing the /θ/ digraph; "nginx" is read "engine-X"; "mongodb" ends in
spelled-out letters. No amount of seq2seq accuracy recovers that, so it is
looked up rather than predicted -- the same reason every production TTS ships a
lexicon.

Values use the eng_to_ipa dialect (see english_inventory) and are verified
against the trained inventory by tests, so a typo here cannot reach the model.
"""

from __future__ import annotations

# Terms whose g2p_en prediction was measured wrong on this checkpoint. Keep the
# key lowercase; the OOV path lowercases before lookup.
TECH_LEXICON: dict[str, str] = {
    # Compound names: the seam misleads spelling-based prediction.
    "github": "gɪthəb",
    "gitlab": "gɪtlæb",
    "pytorch": "paɪtɔrʧ",
    "typescript": "taɪpskrɪpt",
    "frontend": "frəntɛnd",
    "readme": "ridmi",
    "numpy": "nəmpaɪ",
    "huggingface": "həgɪŋfeɪs",
    # Names containing spelled-out letters.
    "nginx": "ɛnʤɪnɛks",
    "mongodb": "mɑŋgoʊdibi",
    "postgresql": "poʊstgrɛskjuɛl",
    "postgres": "poʊstgrɛs",
    "openai": "oʊpəneɪaɪ",
    "onnxruntime": "ɑnɪksrəntaɪm",
    "onnx": "ɑnɪks",
    # Names with a conventional reading no spelling rule predicts.
    "kubernetes": "kubərnɛtiz",
    "jira": "ʤaɪrə",
    "vercel": "vərsɛl",
    "kotlin": "kɑtlɪn",
    "golang": "goʊlæŋ",
    "redis": "rɛdɪs",
}


class LexiconBackend:
    """ForeignG2P serving curated entries only; unknown tokens fall through."""

    def __init__(self, lexicon: dict[str, str] | None = None) -> None:
        source = TECH_LEXICON if lexicon is None else lexicon
        self._lexicon = dict(source)

    def to_ipa(self, token: str) -> str | None:
        return self._lexicon.get(token.lower())
