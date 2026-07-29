"""Shared disposition and retrieval-source contracts for turn routing."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, cast

from soca.core.tool_routing import TurnDisposition

SourceName = Literal["knowledge", "memory"]
SourceProfile = Literal["knowledge", "memory", "both", "neither"]

_SOURCE_NAMES = frozenset({"knowledge", "memory"})


@dataclass(frozen=True)
class RouteSpec:
    name: TurnDisposition
    requires_handler: bool = False
    allows_sources: bool = False


DEFAULT_ROUTE_CATALOG: tuple[RouteSpec, ...] = (
    RouteSpec("direct_tool", requires_handler=True),
    RouteSpec("retrieval_request", allows_sources=True),
    RouteSpec("smalltalk"),
    RouteSpec("out_of_scope"),
    RouteSpec("unresolved"),
)

_ROUTES = {spec.name: spec for spec in DEFAULT_ROUTE_CATALOG}


def source_profile(sources: Iterable[str]) -> SourceProfile:
    selected = frozenset(sources)
    if not selected:
        return "neither"
    if not selected <= _SOURCE_NAMES:
        raise ValueError(f"unknown retrieval source: {sorted(selected - _SOURCE_NAMES)}")
    if selected == {"knowledge"}:
        return "knowledge"
    if selected == {"memory"}:
        return "memory"
    return "both"


def validate_route_fields(
    route: str,
    *,
    handler: str | None,
    sources: Iterable[str],
) -> tuple[TurnDisposition, tuple[str, ...], SourceProfile | None]:
    spec = _ROUTES.get(route)
    if spec is None:
        raise ValueError(f"unknown route: {route}")
    normalized_sources = tuple(sorted(set(sources)))
    if not spec.allows_sources and normalized_sources:
        raise ValueError(f"route {route} cannot select retrieval sources")
    if spec.requires_handler and not handler:
        raise ValueError(f"route {route} requires a handler")
    if not spec.requires_handler and handler is not None:
        raise ValueError(f"route {route} cannot name a handler")
    profile = source_profile(normalized_sources) if spec.allows_sources else None
    return cast(TurnDisposition, route), normalized_sources, profile


__all__ = [
    "DEFAULT_ROUTE_CATALOG",
    "RouteSpec",
    "SourceName",
    "SourceProfile",
    "source_profile",
    "validate_route_fields",
]
