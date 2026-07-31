from __future__ import annotations

# IPA characters eng_to_ipa can emit AND the Valtec checkpoint was trained on
# (measured from enc_p.emb.weight row norms; untrained rows sit at ~0.43).
# See zplan/tts_foreign_g2p_fix_plan.vi.md §2.
TRAINED_ENGLISH_IPA: frozenset[str] = frozenset(
    "abdefghijklmnoprstuvwzæðŋɑɔəɛɪʃʊʒʤʧθ"
)

# d = 0.61: below the ALIVE floor (~1.0) but above the dead baseline (~0.43).
# Kept because (a) it is the dialectally correct target (eng_to_ipa renders
# "the" -> "ðə") and (b) the CMU path already emits it today; excluding it
# here would make the OOV branch stricter than the code already shipping.
WEAKLY_TRAINED: frozenset[str] = frozenset("ð")


def is_trained_ipa(ipa: str) -> bool:
    return bool(ipa) and all(char in TRAINED_ENGLISH_IPA for char in ipa)
