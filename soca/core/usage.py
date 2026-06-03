"""Runtime usage/telemetry data model.

Pure, immutable data + aggregation only. NO Rich/console imports — rendering
lives in the app layer (`soca/app/usage_view.py`). Builders are duck-typed so
this module never imports `RuntimeResult`/`StreamingEvent` (avoids a cycle and
keeps the model reusable for both the text and voice paths).

Numbers come from the real runtime (token counts, TTFT, tok/s, stage latencies,
TTFA).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class LLMUsage:
    """Per-call LLM telemetry, normalized across streaming and non-streaming."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0
    total_latency_ms: float = 0.0
    tokens_per_second: float = 0.0

    @classmethod
    def from_llm_result(cls, result: Any | None) -> LLMUsage | None:
        """Build from a non-streaming ``LLMResult`` (or anything with the same fields)."""
        if result is None:
            return None
        return cls(
            prompt_tokens=_as_int(getattr(result, "n_prompt_tokens", 0)),
            completion_tokens=_as_int(getattr(result, "n_completion_tokens", 0)),
            ttft_ms=_as_float(getattr(result, "ttft_ms", 0.0)),
            total_latency_ms=_as_float(getattr(result, "total_latency_ms", 0.0)),
            tokens_per_second=_as_float(getattr(result, "tokens_per_second", 0.0)),
        )


@dataclass(frozen=True)
class TurnUsage:
    """Usage for one turn. Voice-only fields stay ``None`` for text turns."""

    route: str
    blocked: bool = False
    used_tool: bool = False
    used_llm: bool = False
    llm: LLMUsage | None = None
    asr_latency_ms: float | None = None
    runtime_latency_ms: float | None = None
    tts_first_chunk_latency_ms: float | None = None
    ttfa_ms: float | None = None
    total_turn_latency_ms: float | None = None
    tts_chunks: int | None = None

    @classmethod
    def from_runtime_result(cls, result: Any) -> TurnUsage:
        """Build a text-turn usage record from a ``RuntimeResult`` (duck-typed)."""
        trace = getattr(result, "trace", None)
        stages = dict(getattr(trace, "stage_latencies_ms", {}) or {})
        return cls(
            route=_route_value(result),
            blocked=bool(getattr(result, "blocked", False)),
            used_tool=bool(getattr(trace, "used_tool", False)),
            used_llm=bool(getattr(trace, "used_llm", False)),
            llm=getattr(result, "usage", None),
            runtime_latency_ms=stages.get("llm"),
        )

    @classmethod
    def from_voice(
        cls,
        *,
        route: str,
        blocked: bool,
        llm: LLMUsage | None,
        stage_latencies_ms: dict[str, float] | None = None,
        total_turn_latency_ms: float | None = None,
        first_tts_latency_ms: float | None = None,
        ttfa_ms: float | None = None,
        tts_chunks: int = 0,
    ) -> TurnUsage:
        """Aggregate one voice turn from pipeline streaming-event metadata."""
        stages = dict(stage_latencies_ms or {})
        return cls(
            route=route,
            blocked=blocked,
            used_tool=False,
            used_llm=llm is not None,
            llm=llm,
            asr_latency_ms=stages.get("asr"),
            runtime_latency_ms=stages.get("llm"),
            tts_first_chunk_latency_ms=first_tts_latency_ms,
            ttfa_ms=ttfa_ms,
            total_turn_latency_ms=total_turn_latency_ms,
            tts_chunks=tts_chunks or None,
        )


def _route_value(result: Any) -> str:
    route = getattr(result, "route", None)
    return str(getattr(route, "value", route) or "")


@dataclass(frozen=True)
class SessionUsage:
    """Immutable accumulator of per-turn usage for a chat/voice session."""

    turns: tuple[TurnUsage, ...] = ()

    def add(self, turn: TurnUsage) -> SessionUsage:
        return SessionUsage(turns=(*self.turns, turn))

    @property
    def total_turns(self) -> int:
        return len(self.turns)

    @property
    def llm_turns(self) -> int:
        return sum(1 for turn in self.turns if turn.llm is not None)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(turn.llm.prompt_tokens for turn in self.turns if turn.llm)

    @property
    def total_completion_tokens(self) -> int:
        return sum(turn.llm.completion_tokens for turn in self.turns if turn.llm)

    @property
    def mean_tokens_per_second(self) -> float:
        rates = [
            turn.llm.tokens_per_second
            for turn in self.turns
            if turn.llm and turn.llm.tokens_per_second > 0
        ]
        return sum(rates) / len(rates) if rates else 0.0

    @property
    def mean_ttft_ms(self) -> float:
        values = [turn.llm.ttft_ms for turn in self.turns if turn.llm and turn.llm.ttft_ms > 0]
        return sum(values) / len(values) if values else 0.0


__all__ = ["LLMUsage", "TurnUsage", "SessionUsage"]
