from __future__ import annotations

from typing import Any

CURRENT_PROTOCOL_VERSION = 2
SUPPORTED_PROTOCOL_VERSIONS = (1, 2)


def adapt_legacy_command(command: dict[str, Any]) -> dict[str, Any]:
    """Normalize a v1 command without changing its meaning."""
    normalized = dict(command)
    normalized.pop("protocol_version", None)
    return normalized


def protocol_hello(*, profile: str, no_model: bool, stack: dict[str, Any]) -> dict[str, Any]:
    return {
        "event": "hello",
        "version": CURRENT_PROTOCOL_VERSION,
        "protocol_version": CURRENT_PROTOCOL_VERSION,
        "supported_versions": list(SUPPORTED_PROTOCOL_VERSIONS),
        "profile": profile,
        "no_model": no_model,
        "stack": stack,
    }
