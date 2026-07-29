"""Versioned, offline-fitted routing calibration artifact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from soca.core.route_catalog import SourceProfile, source_profile
from soca.core.tool_routing import TurnDisposition

_ROUTES: tuple[TurnDisposition, ...] = (
    "direct_tool",
    "retrieval_request",
    "smalltalk",
    "out_of_scope",
    "unresolved",
)
_SOURCES = ("knowledge", "memory")


@dataclass(frozen=True)
class CalibrationArtifact:
    version: int
    encoder_id: str
    aggregation: str
    route_thresholds: dict[str, float]
    route_margin: float
    source_thresholds: dict[str, float]
    source_margin: float
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError("unsupported calibration artifact version")
        if not self.encoder_id.strip() or not self.aggregation.strip():
            raise ValueError("calibration encoder metadata is required")
        _validate_thresholds(self.route_thresholds, _ROUTES, "route")
        _validate_thresholds(self.source_thresholds, _SOURCES, "source")
        _validate_probability(self.route_margin, "route margin")
        _validate_probability(self.source_margin, "source margin")

    def route(self, scores: dict[str, float]) -> tuple[TurnDisposition, str | None, float | None]:
        ranked = sorted(
            ((route, float(scores.get(route, float("-inf")))) for route in _ROUTES),
            key=lambda item: (-item[1], item[0]),
        )
        top, top_score = ranked[0]
        runner_up = ranked[1][0] if len(ranked) > 1 else None
        margin = top_score - ranked[1][1] if runner_up is not None else None
        threshold = self.route_thresholds[top]
        if top_score < threshold or (margin is not None and margin < self.route_margin):
            return "unresolved", runner_up, margin
        return cast(TurnDisposition, top), runner_up, margin

    def select_sources(self, scores: dict[str, float]) -> tuple[SourceProfile, tuple[str, ...]]:
        available = {source: float(scores.get(source, float("-inf"))) for source in _SOURCES}
        best_score = max(available.values())
        selected = tuple(
            source
            for source in _SOURCES
            if available[source] >= self.source_thresholds[source]
            and best_score - available[source] <= self.source_margin
        )
        return source_profile(selected), selected

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_version": self.version,
            "encoder_id": self.encoder_id,
            "aggregation": self.aggregation,
            "route_thresholds": dict(self.route_thresholds),
            "route_margin": self.route_margin,
            "source_thresholds": dict(self.source_thresholds),
            "source_margin": self.source_margin,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CalibrationArtifact:
        return cls(
            version=int(payload.get("artifact_version", 0)),
            encoder_id=str(payload.get("encoder_id", "")),
            aggregation=str(payload.get("aggregation", "")),
            route_thresholds={str(k): float(v) for k, v in dict(payload["route_thresholds"]).items()},
            route_margin=float(payload["route_margin"]),
            source_thresholds={str(k): float(v) for k, v in dict(payload["source_thresholds"]).items()},
            source_margin=float(payload["source_margin"]),
            metadata=dict(payload.get("metadata", {})),
        )


def _validate_probability(value: float, label: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")


def _validate_thresholds(values: dict[str, float], expected: tuple[str, ...], label: str) -> None:
    if set(values) != set(expected):
        raise ValueError(f"{label} thresholds must cover {expected}")
    for key, value in values.items():
        _validate_probability(float(value), f"{label} threshold {key}")


__all__ = ["CalibrationArtifact"]
