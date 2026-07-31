"""Product, brand and project names.

These break English spelling rules on purpose, so a spelling-based predictor
cannot reach them: "github" is git+hub rather than a word containing the /θ/
digraph, and "nginx" is spoken "engine-X". Names the predictor already handles
(netflix, figma, heroku, firebase, kotlin) are intentionally absent.
"""

from __future__ import annotations

TOOLS_LEXICON: dict[str, str] = {
    # Compound names whose seam misleads the predictor.
    "frontend": "frəntɛnd",
    "github": "gɪthəb",
    "gitlab": "gɪtlæb",
    "huggingface": "həgɪŋfeɪs",
    "numpy": "nəmpaɪ",
    "pytorch": "paɪtɔrʧ",
    "readme": "ridmi",
    "typescript": "taɪpskrɪpt",
    # Names ending in spelled-out letters.
    "coreml": "kɔrɛmɛl",
    "mongodb": "mɑŋgoʊdibi",
    "onnx": "ɑnɪks",
    "onnxruntime": "ɑnɪksrəntaɪm",
    "openai": "oʊpəneɪaɪ",
    "postgres": "poʊstgrɛs",
    "postgresql": "poʊstgrɛskjuɛl",
    # Conventional readings no spelling rule predicts.
    "jira": "ʤaɪrə",
    "kubernetes": "kubərnɛtiz",
    "nginx": "ɛnʤɪnɛks",
    "qodo": "koʊdoʊ",
    "redis": "rɛdɪs",
    "soca": "soʊkɑ",
    "vercel": "vərsɛl",
}
