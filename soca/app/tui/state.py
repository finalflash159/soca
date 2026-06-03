from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from soca.core.profiles import DEFAULT_VOICE_RUNTIME_PROFILE_KEY
from soca.core.usage import SessionUsage, TurnUsage

TuiMode = Literal["status", "chat", "voice"]


@dataclass
class TuiState:
    mode: TuiMode = "status"
    profile: str = DEFAULT_VOICE_RUNTIME_PROFILE_KEY
    runtime_state: str = "idle"
    session_usage: SessionUsage = field(default_factory=SessionUsage)
    last_turn_usage: TurnUsage | None = None
    trace_enabled: bool = False


__all__ = ["TuiMode", "TuiState"]
