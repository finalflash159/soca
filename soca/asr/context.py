from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol


def normalize_context_text(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("ASR context values must be strings")
    return " ".join(unicodedata.normalize("NFC", value).split())


def approximate_context_tokens(value: str) -> int:
    if not value:
        return 0
    return (len(value.encode("utf-8")) + 3) // 4


@dataclass(frozen=True, slots=True)
class ASRContextLimits:
    max_chars: int = 1_024
    max_approximate_tokens: int = 256
    max_terms: int = 64

    def __post_init__(self) -> None:
        for name, value in (
            ("max_chars", self.max_chars),
            ("max_approximate_tokens", self.max_approximate_tokens),
            ("max_terms", self.max_terms),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, int]:
        return {
            "max_chars": self.max_chars,
            "max_approximate_tokens": self.max_approximate_tokens,
            "max_terms": self.max_terms,
        }

    @property
    def policy_digest(self) -> str:
        payload = {
            "limits": self.to_dict(),
            "normalization": "unicode_nfc_whitespace_v1",
            "ordering": "priority_provenance_value_v1",
            "schema_version": 1,
        }
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ASRContextSourceRecord:
    value: str
    provenance: str
    priority: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise TypeError("ASR context record value must be a string")
        if not isinstance(self.provenance, str):
            raise TypeError("ASR context provenance must be a string")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise TypeError("ASR context priority must be an integer")
        normalized_provenance = normalize_context_text(self.provenance)
        if not normalized_provenance:
            raise ValueError("ASR context provenance must not be empty")


@dataclass(frozen=True, slots=True)
class ASRContextEntry:
    value: str
    provenance: str
    priority: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "value": self.value,
            "provenance": self.provenance,
            "priority": self.priority,
        }


@dataclass(frozen=True, slots=True)
class ASRContextSnapshot:
    text: str
    entries: tuple[ASRContextEntry, ...]
    limits: ASRContextLimits
    approximate_tokens: int
    digest: str

    @property
    def policy_digest(self) -> str:
        return self.limits.policy_digest

    @property
    def provenances(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(entry.provenance for entry in self.entries))

    @property
    def term_count(self) -> int:
        return len(self.entries)

    @property
    def is_empty(self) -> bool:
        return not self.entries


class ASRContextProvider(Protocol):
    def snapshot(self) -> ASRContextSnapshot: ...


ASRContextSourceLoader = Callable[[], Iterable[ASRContextSourceRecord]]


class ASRContextBuilder:
    def __init__(self, limits: ASRContextLimits | None = None) -> None:
        self._limits = limits or ASRContextLimits()

    @property
    def limits(self) -> ASRContextLimits:
        return self._limits

    def build(self, records: Iterable[ASRContextSourceRecord]) -> ASRContextSnapshot:
        normalized = self._normalize_records(records)
        selected: list[ASRContextEntry] = []
        text = ""

        for entry in normalized:
            if len(selected) >= self._limits.max_terms:
                break
            candidate = ", ".join((*[item.value for item in selected], entry.value))
            if len(candidate) > self._limits.max_chars:
                continue
            if approximate_context_tokens(candidate) > self._limits.max_approximate_tokens:
                continue
            selected.append(entry)
            text = candidate

        entries = tuple(selected)
        approximate_tokens = approximate_context_tokens(text)
        canonical = _canonical_snapshot(text, entries, self._limits, approximate_tokens)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return ASRContextSnapshot(
            text=text,
            entries=entries,
            limits=self._limits,
            approximate_tokens=approximate_tokens,
            digest=digest,
        )

    @staticmethod
    def _normalize_records(
        records: Iterable[ASRContextSourceRecord],
    ) -> tuple[ASRContextEntry, ...]:
        unique: dict[str, ASRContextEntry] = {}
        for record in records:
            if not isinstance(record, ASRContextSourceRecord):
                raise TypeError("ASR context sources must be ASRContextSourceRecord values")
            value = normalize_context_text(record.value)
            if not value:
                continue
            provenance = normalize_context_text(record.provenance)
            identity = value.casefold()
            candidate = ASRContextEntry(
                value=value,
                provenance=provenance,
                priority=record.priority,
            )
            incumbent = unique.get(identity)
            if incumbent is None or _entry_sort_key(candidate) < _entry_sort_key(incumbent):
                unique[identity] = candidate
        return tuple(sorted(unique.values(), key=_entry_sort_key))


@dataclass(frozen=True, slots=True)
class DynamicASRContextProvider:
    source_loader: ASRContextSourceLoader
    builder: ASRContextBuilder

    def snapshot(self) -> ASRContextSnapshot:
        return self.builder.build(self.source_loader())


def _entry_sort_key(entry: ASRContextEntry) -> tuple[int, str, str, str, str]:
    return (
        -entry.priority,
        entry.provenance.casefold(),
        entry.provenance,
        entry.value.casefold(),
        entry.value,
    )


def _canonical_snapshot(
    text: str,
    entries: tuple[ASRContextEntry, ...],
    limits: ASRContextLimits,
    approximate_tokens: int,
) -> str:
    payload = {
        "approximate_tokens": approximate_tokens,
        "entries": [entry.to_dict() for entry in entries],
        "limits": limits.to_dict(),
        "schema_version": 1,
        "text": text,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


__all__ = [
    "ASRContextBuilder",
    "ASRContextEntry",
    "ASRContextLimits",
    "ASRContextProvider",
    "ASRContextSnapshot",
    "ASRContextSourceLoader",
    "ASRContextSourceRecord",
    "DynamicASRContextProvider",
    "approximate_context_tokens",
    "normalize_context_text",
]
