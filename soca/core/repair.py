"""Conversation-repair vocabulary and selection (plan §4, §7, R1).

This module turns a *technical* reason (e.g. ``no_speech``, ``low_confidence``)
into a *user-facing* follow-up line, picked from a controlled catalog of natural
Vietnamese variants instead of a single hardcoded string. Selection is a pure
function (testable, no-repeat aware); the small per-session :class:`RepairState`
tracks attempt escalation and recently used prompts.

Design rules (see plan §7, §16):
  - không dùng LLM để sinh câu repair;
  - không lặp lại prompt vừa dùng nếu còn biến thể khác;
  - guardrail/safety dùng ít biến thể, không hài hóa;
  - text được TTS đọc nên không markdown/emoji.
"""

from __future__ import annotations

import tomllib
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from random import Random

_CATALOG_PATH = Path(__file__).with_name("repair_prompts.vi.toml")

# Ultimate fallback if the catalog is missing a slot — never crash the voice loop.
DEFAULT_REPAIR_TEXT = "Mình chưa nghe rõ. Bạn nói lại giúp mình nha."


class RepairKind(Enum):
    NO_INPUT = "no_input"
    UNCERTAIN_INPUT = "uncertain_input"
    NO_MATCH = "no_match"
    OUT_OF_SCOPE = "out_of_scope"
    GUARDRAIL_BLOCKED = "guardrail_blocked"
    TOOL_FAILED = "tool_failed"
    KNOWLEDGE_MISS = "knowledge_miss"
    TTS_FAILED = "tts_failed"
    SESSION_INACTIVE = "session_inactive"


class RepairAction(Enum):
    REPROMPT = "reprompt"
    CONTEXTUAL_REPROMPT = "contextual_reprompt"
    CLARIFY = "clarify"
    REDIRECT = "redirect"
    HANDOVER_TO_CHAT = "handover_to_chat"
    NO_REPLY_FOLLOWUP = "no_reply_followup"
    NO_REPLY_GUIDANCE = "no_reply_guidance"
    SLEEP_VOICE = "sleep_voice"
    SHOW_TEXT = "show_text"


@dataclass(frozen=True)
class RepairSlot:
    """One catalog entry: a list of interchangeable variants for a situation."""

    kind: RepairKind
    slot: str
    action: RepairAction
    variants: tuple[str, ...]


@dataclass(frozen=True)
class RepairChoice:
    """A picked repair line plus the metadata needed for UI/debug + no-repeat."""

    text: str
    prompt_id: str
    kind: RepairKind
    action: RepairAction


class RepairCatalog:
    """Loaded set of repair slots with pure no-repeat selection."""

    def __init__(self, slots: dict[str, RepairSlot]) -> None:
        self._slots = dict(slots)

    @classmethod
    def from_toml(cls, path: Path | None = None) -> RepairCatalog:
        data = tomllib.loads((path or _CATALOG_PATH).read_text(encoding="utf-8"))
        slots: dict[str, RepairSlot] = {}
        for kind_name, slot_map in data.items():
            kind = RepairKind(kind_name)
            for slot_name, body in slot_map.items():
                slots[f"{kind_name}.{slot_name}"] = RepairSlot(
                    kind=kind,
                    slot=slot_name,
                    action=RepairAction(body["action"]),
                    variants=tuple(body["variants"]),
                )
        return cls(slots)

    def has(self, kind: RepairKind, slot: str) -> bool:
        return f"{kind.value}.{slot}" in self._slots

    def select(
        self,
        kind: RepairKind,
        slot: str,
        *,
        rng: Random,
        recent_ids: object = (),
    ) -> RepairChoice:
        """Pick a variant for ``kind.slot``, avoiding ``recent_ids`` when possible.

        Pure: no mutation. The caller owns the recent-id history and the rng.
        """
        recent = set(recent_ids)  # type: ignore[arg-type]
        key = f"{kind.value}.{slot}"
        repair_slot = self._slots.get(key)
        if repair_slot is None or not repair_slot.variants:
            return RepairChoice(
                text=DEFAULT_REPAIR_TEXT,
                prompt_id=f"{key}#default",
                kind=kind,
                action=RepairAction.REPROMPT,
            )

        indexed = list(enumerate(repair_slot.variants))
        fresh = [(i, v) for i, v in indexed if f"{key}#{i}" not in recent]
        index, text = rng.choice(fresh or indexed)
        return RepairChoice(
            text=text,
            prompt_id=f"{key}#{index}",
            kind=repair_slot.kind,
            action=repair_slot.action,
        )

    def validate(self) -> list[str]:
        """Lint the catalog: non-empty variants, no markdown/emoji in speech text."""
        errors: list[str] = []
        for key, slot in self._slots.items():
            if not slot.variants:
                errors.append(f"{key}: no variants")
            for variant in slot.variants:
                if not variant.strip():
                    errors.append(f"{key}: empty variant")
                if any(token in variant for token in ("**", "##", "- ", "`")):
                    errors.append(f"{key}: markdown-like token in speech text: {variant!r}")
        return errors


@dataclass
class RepairState:
    """Per-session repair state (not persisted). Mutable by design — it tracks an
    escalation ladder across consecutive failed turns."""

    no_input_attempts: int = 0
    recent_prompt_ids: deque[str] = field(default_factory=lambda: deque(maxlen=8))

    def reset(self) -> None:
        """Called after a successful turn — clears the escalation ladder."""
        self.no_input_attempts = 0


# technical ASR reason -> repair kind. Unknown reasons fall back to no_input.
_UNCERTAIN_MARKERS = ("low_confidence", "compression", "hallucinat", "repetition")


def kind_for_reason(rejection_reason: str) -> RepairKind:
    reason = (rejection_reason or "").lower()
    if any(marker in reason for marker in _UNCERTAIN_MARKERS):
        return RepairKind.UNCERTAIN_INPUT
    return RepairKind.NO_INPUT


def _slot_for_attempt(kind: RepairKind, attempt: int) -> str:
    """Escalation ladder: gentle reprompt -> contextual -> handover."""
    if kind == RepairKind.NO_INPUT:
        if attempt <= 1:
            return "attempt_1"
        if attempt == 2:
            return "attempt_2"
        return "handover"
    if kind == RepairKind.UNCERTAIN_INPUT:
        return "attempt_1" if attempt <= 1 else "attempt_2"
    return "attempt_1"


def plan_repair(
    catalog: RepairCatalog,
    *,
    rejection_reason: str,
    state: RepairState,
    rng: Random,
) -> RepairChoice:
    """Plan one repair line for an empty/rejected ASR turn and update ``state``.

    Increments the attempt counter (escalation), picks a fresh variant, and
    records the chosen prompt id so the next call avoids repeating it.
    """
    kind = kind_for_reason(rejection_reason)
    state.no_input_attempts += 1
    slot = _slot_for_attempt(kind, state.no_input_attempts)
    choice = catalog.select(
        kind,
        slot,
        rng=rng,
        recent_ids=tuple(state.recent_prompt_ids),
    )
    state.recent_prompt_ids.append(choice.prompt_id)
    return choice


@dataclass(frozen=True)
class RepairTimings:
    """No-reply / inactivity thresholds (plan §6.1). Milliseconds of silence
    while SoCa is waiting, not the recorder endpoint silence."""

    no_reply_1_at_ms: int = 45_000
    no_reply_2_at_ms: int = 120_000
    sleep_voice_at_ms: int = 300_000
    passive_sleep_at_ms: int = 300_000


_DEFAULT_TIMINGS = RepairTimings()


def plan_no_reply(
    silence_ms: float,
    *,
    expects_response: bool,
    attempts_fired: int,
    timings: RepairTimings | None = None,
) -> str | None:
    """Pure no-reply ladder: which ``session_inactive`` slot to fire, or None.

    When SoCa is not waiting on the user (``expects_response=False``), passive
    silence never speaks a follow-up — it only sleeps after a long idle. When
    SoCa *is* waiting, it escalates gently: a soft follow-up, then guidance, then
    sleep — each fired at most once per idle stretch (tracked by ``attempts_fired``).
    """
    timings = timings or _DEFAULT_TIMINGS
    if not expects_response:
        return "sleep" if silence_ms >= timings.passive_sleep_at_ms else None
    if silence_ms >= timings.sleep_voice_at_ms:
        return "sleep"
    if silence_ms >= timings.no_reply_2_at_ms and attempts_fired < 2:
        return "no_reply_2"
    if silence_ms >= timings.no_reply_1_at_ms and attempts_fired < 1:
        return "no_reply_1"
    return None


_DEFAULT_CATALOG: RepairCatalog | None = None


def default_repair_catalog() -> RepairCatalog:
    """Lazily-loaded shared catalog used by the production voice pipeline."""
    global _DEFAULT_CATALOG
    if _DEFAULT_CATALOG is None:
        _DEFAULT_CATALOG = RepairCatalog.from_toml()
    return _DEFAULT_CATALOG


__all__ = [
    "DEFAULT_REPAIR_TEXT",
    "RepairAction",
    "RepairCatalog",
    "RepairChoice",
    "RepairKind",
    "RepairSlot",
    "RepairState",
    "RepairTimings",
    "default_repair_catalog",
    "kind_for_reason",
    "plan_no_reply",
    "plan_repair",
]
