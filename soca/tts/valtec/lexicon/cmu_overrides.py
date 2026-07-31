"""Words CMUdict contains but pronounces wrongly for this domain.

Unlike every other lexicon file, these shadow a dictionary entry, so each one
needs evidence that the dictionary is wrong rather than merely surprising.
Tests assert each key really is in CMU and that the override differs from it,
so an entry that upstream later fixes shows up as a failure instead of
silently freezing a stale reading.
"""

from __future__ import annotations

CMU_OVERRIDE_LEXICON: dict[str, str] = {
    # CMU gives kæˈʃeɪ, the "cachet" reading. Its own inflected forms disagree
    # (cached -> kæʃt, caching -> ˈkæʃɪŋ), which is the tell that the base
    # entry is the anomaly.
    "cache": "kæʃ",
}
