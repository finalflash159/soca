from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASET_CLASSES = frozenset(
    {"demo_smoke", "unit_fixture", "public_screening", "sanitized_benchmark", "private_release"}
)
QUALITY_DATASET_CLASSES = frozenset({"public_screening", "sanitized_benchmark", "private_release"})
SPLITS = frozenset({"train", "dev", "test", "challenge", "release"})
SUITE_KINDS = frozenset({"regression", "capability"})
TERMINAL_OUTCOMES = frozenset(
    {
        "achieved",
        "needs_clarification",
        "insufficient_evidence",
        "safe_failure",
        "budget_exhausted",
        "cancelled",
        "system_failure",
    }
)


@dataclass(frozen=True)
class RemediationCase:
    case_id: str
    suite_kind: str
    dataset_class: str
    split: str
    family: str
    category: str
    turns: tuple[str, ...]
    expected_goal: str
    expected_terminal: str
    expected_sources: tuple[str, ...]
    expected_tools: tuple[str, ...]
    expected_citations: tuple[str, ...]
    audit_items: tuple[str, ...]
    provenance: str
    metadata: dict[str, Any]


def _reject_demo_derivative(payload: dict[str, Any], path: Path, line_no: int) -> None:
    serialized = json.dumps(payload, ensure_ascii=False).lower()
    forbidden = ("knowledge_demo_vault", "demo_vault", "derived_from_demo")
    if any(marker in serialized for marker in forbidden):
        raise ValueError(f"{path}:{line_no} is derived from demo data")


def load_cases(path: Path, *, quality_suite: bool = False) -> tuple[RemediationCase, ...]:
    cases: list[RemediationCase] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no} must be an object")
            _reject_demo_derivative(payload, path, line_no)
            dataset_class = payload.get("dataset_class")
            split = payload.get("split")
            suite_kind = payload.get("suite_kind")
            family = payload.get("family")
            case_id = payload.get("id")
            turns = payload.get("turns")
            expected_terminal = payload.get("expected_terminal")
            provenance = payload.get("provenance")
            audit_items = payload.get("audit_items")
            if (
                not isinstance(case_id, str)
                or not case_id.strip()
                or case_id in seen
                or suite_kind not in SUITE_KINDS
                or dataset_class not in DATASET_CLASSES
                or (quality_suite and dataset_class not in QUALITY_DATASET_CLASSES)
                or split not in SPLITS
                or not isinstance(family, str)
                or not family.strip()
                or not isinstance(turns, list)
                or not turns
                or not all(isinstance(turn, str) and turn.strip() for turn in turns)
                or expected_terminal not in TERMINAL_OUTCOMES
                or not isinstance(audit_items, list)
                or not audit_items
                or not all(isinstance(item, str) and item.strip() for item in audit_items)
                or not isinstance(provenance, str)
                or not provenance.strip()
            ):
                raise ValueError(f"{path}:{line_no} has invalid remediation case")
            expected_sources = payload.get("expected_sources", [])
            expected_tools = payload.get("expected_tools", [])
            expected_citations = payload.get("expected_citations", [])
            expected_values = (expected_sources, expected_tools, expected_citations)
            if not all(isinstance(values, list) for values in expected_values):
                raise ValueError(f"{path}:{line_no} expected values must be JSON lists")
            if not all(
                isinstance(value, str)
                for values in expected_values
                for value in values
            ):
                raise ValueError(f"{path}:{line_no} expected values must be strings")
            seen.add(case_id)
            cases.append(
                RemediationCase(
                    case_id=case_id,
                    suite_kind=suite_kind,
                    dataset_class=dataset_class,
                    split=split,
                    family=family.strip(),
                    category=str(payload.get("category", "unspecified")),
                    turns=tuple(turns),
                    expected_goal=str(payload.get("expected_goal", "")),
                    expected_terminal=expected_terminal,
                    expected_sources=tuple(expected_sources),
                    expected_tools=tuple(expected_tools),
                    expected_citations=tuple(expected_citations),
                    audit_items=tuple(audit_items),
                    provenance=provenance,
                    metadata=dict(payload.get("metadata", {})),
                )
            )
    if not cases:
        raise ValueError(f"{path}: empty dataset")
    return tuple(cases)


def assert_quality_eligible(cases: tuple[RemediationCase, ...]) -> None:
    """Fail closed when a quality/release collection contains demo/smoke rows."""

    ineligible = [case.case_id for case in cases if case.dataset_class not in QUALITY_DATASET_CLASSES]
    if ineligible:
        raise ValueError("quality/release suite contains non-quality cases: " + ", ".join(ineligible))


def assert_no_family_leakage(cases: tuple[RemediationCase, ...]) -> None:
    splits_by_family: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        splits_by_family[case.family].add(case.split)
    leaked = sorted(
        family
        for family, splits in splits_by_family.items()
        if len(splits) > 1
    )
    if leaked:
        raise ValueError(
            "paraphrase families cross dataset splits: " + ", ".join(leaked)
        )


__all__ = [
    "DATASET_CLASSES",
    "QUALITY_DATASET_CLASSES",
    "RemediationCase",
    "SUITE_KINDS",
    "assert_no_family_leakage",
    "assert_quality_eligible",
    "load_cases",
]
