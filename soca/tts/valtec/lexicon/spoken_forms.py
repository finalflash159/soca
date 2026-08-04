"""Vietnamese spoken forms for technical terms with unstable prosody.

These forms are applied only to Valtec's private speech input. The assistant
text shown in the UI remains unchanged. Native Vietnamese G2P is intentional:
the Valtec checkpoint has stable tone behavior for these syllables, while the
foreign IPA path made the terms sound split or unnaturally high-pitched.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TechnicalSpeechForm:
    """Measured speech-only rewrite and pacing for one technical term."""

    spoken: str
    duration_scale: float = 1.0
    pause_after: bool = False


TECHNICAL_SPEECH_FORMS: dict[str, TechnicalSpeechForm] = {
    # Expanding the initialism avoids the checkpoint's unstable letter-name
    # reading ("eo eo em") and gives the listener the actual concept.
    "llm": TechnicalSpeechForm("large language model"),
    # The checkpoint splits the foreign consonant cluster; native syllables
    # keep the compound together and leave enough time for both parts.
    "softmax": TechnicalSpeechForm("sóp mác", duration_scale=1.1),
    "cosine": TechnicalSpeechForm("cô sai", duration_scale=1.1),
    # The English G2P output is unstable for this short word in Vietnamese
    # sentences; a Vietnamese respelling keeps the two syllables distinct.
    "paper": TechnicalSpeechForm("pây pờ", duration_scale=1.1),
    # The extra syllable time keeps the three-part borrowed reading coherent.
    "embedding": TechnicalSpeechForm("em bê đinh", duration_scale=1.1),
    "api": TechnicalSpeechForm("ây pi ai", duration_scale=1.1),
    # The checkpoint inserts an audible break in the /paɪplaɪn/ cluster;
    # Vietnamese syllables keep the compound continuous without a pause.
    "pipeline": TechnicalSpeechForm("pai lain", duration_scale=1.15),
    # Keep the compound technical terms as deliberate Vietnamese syllables;
    # the foreign path tends to swallow the unstressed middle syllables.
    "long-context": TechnicalSpeechForm("long con téc", duration_scale=1.15),
    "rope": TechnicalSpeechForm("rô pê", duration_scale=1.1),
    "scaling": TechnicalSpeechForm("sờ cê lình", duration_scale=1.1),
    "activation": TechnicalSpeechForm("ắc ti vây sần", duration_scale=1.15),
    "sparsity": TechnicalSpeechForm("sờ pác si ti", duration_scale=1.1),
    "interpretability": TechnicalSpeechForm(
        "in tơ pờ rơ tơ bi li ti", duration_scale=1.2
    ),
    "factuality": TechnicalSpeechForm("phác chu a li ti", duration_scale=1.15),
    "recompute": TechnicalSpeechForm("ri cầm piut", duration_scale=1.1),
    "err_connection_reset": TechnicalSpeechForm(
        "lỗi kết nối bị ngắt", duration_scale=1.1
    ),
    # The foreign IPA output is distorted by the Vietnamese checkpoint. Native
    # syllables make the borrowed Vietnamese reading stable; the pause prevents
    # the following initialism from collapsing into the final consonant.
    "remote": TechnicalSpeechForm("ri mốt", duration_scale=1.15, pause_after=True),
    "transformer": TechnicalSpeechForm(
        "trăn phơ mơ", duration_scale=1.15, pause_after=True
    ),
}
