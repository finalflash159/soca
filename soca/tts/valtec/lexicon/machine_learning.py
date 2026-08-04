"""Machine-learning vocabulary g2p_en mispredicts.

Only terms whose measured foreign-G2P prediction was wrong appear here. Terms
that need native Vietnamese prosody are kept in ``spoken_forms.py`` instead.
Words the predictor already handles (embedding, tokenizer, encoder,
optimizer, quantization, classifier, ablation, tensor) are deliberately absent
so upstream improvements are not frozen behind a stale hand-written entry.
"""

from __future__ import annotations

MACHINE_LEARNING_LEXICON: dict[str, str] = {
    # g2p_en: ɔtɔrgɛsɪv -- drops the "regressive" stem entirely.
    "autoregressive": "ɔtoʊrɪgrɛsɪv",
    # g2p_en: ɛnteɪməlt -- metathesis of the -ment suffix.
    "entailment": "ɛnteɪlmənt",
    # g2p_en: graʊndɪndəs -- doubles the /n/.
    "groundedness": "graʊndɪdnəs",
    # g2p_en: lɑʤɪt -- "lah-jit" rather than the log- stem.
    "logit": "loʊʤɪt",
    "logits": "loʊʤɪts",
    # g2p_en: sɔftmæsts -- loses the /k/.
    "softmaxed": "sɔftmækst",
    # g2p_en: tɑkənəzeɪʃən -- "tock-" instead of the "token" stem.
    "tokenization": "toʊkənəzeɪʃən",
}
