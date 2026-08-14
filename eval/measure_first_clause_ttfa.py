"""Controlled A/B: first_clause ON vs OFF, holding the LLM token stream constant.

Method: for each transcript, run the REAL runtime once and capture the LLM token
stream WITH per-token arrival times. Then replay that exact stream (same tokens,
same inter-token delays via time.sleep) through the runtime twice -- first_clause
ON and OFF -- and measure wall-clock time to the FIRST 'sentence' event. Because
both replays see identical tokens+timing, the delta isolates the flush-point effect
of first-clause. TTS synth latency of each first chunk (real Valtec) is added to get
a tts_ready_ttfa estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.result_io import make_eval_artifact_metadata, write_json_artifact
from soca.core import AssistantRuntime, RuntimeOptions
from soca.core.turn import RuntimeResult
from soca.llm import LocalLlamaCppLLM
from soca.llm.base import LLMResult
from soca.tts import create_tts_engine

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "eval" / "results" / "first_clause_ttfa_controlled.json"
MODEL_ID = "arcee_vylinh_3b_q4_k_m"
VOICE_ID = "NF"
TRANSCRIPTS = (
    "Chào bạn, bạn là ai vậy?",
    "Giúp tôi kiểm tra lại cấu hình một chút nhé.",
    "Bạn có thể tóm tắt ngắn gọn giúp tôi không?",
    "Hôm nay tôi nên ăn gì để đủ chất đạm?",
    "Kể cho tôi nghe một chút về bản thân bạn đi.",
    "Bạn thấy trời hôm nay thế nào?",
    "Cho tôi một lời khuyên để ngủ ngon hơn nhé.",
    "Giải thích giúp tôi vì sao nên uống đủ nước.",
)


@dataclass(frozen=True)
class Capture:
    timed_tokens: tuple[tuple[float, str], ...]
    terminal: RuntimeResult


@dataclass(frozen=True)
class FlushMeasurement:
    elapsed_ms: float
    text: str
    terminal_route: str


class _ReplayLLM:
    """Yields a captured token stream, reproducing the original per-token delays."""

    def __init__(self, timed_tokens: list[tuple[float, str]]) -> None:
        self._timed = timed_tokens

    def generate(self, user_msg: str, **kwargs) -> LLMResult:
        text = "".join(t for _, t in self._timed)
        return LLMResult(
            text=text,
            prompt=user_msg,
            n_prompt_tokens=1,
            n_completion_tokens=len(self._timed),
            ttft_ms=1.0,
            total_latency_ms=2.0,
            tokens_per_second=1.0,
        )

    def generate_stream(self, user_msg: str, **kwargs) -> Iterator[str]:
        del user_msg, kwargs
        base = time.perf_counter()
        for target_t, token in self._timed:
            now = time.perf_counter() - base
            if target_t > now:
                time.sleep(target_t - now)
            yield token


def _capture(runtime: AssistantRuntime, transcript: str) -> Capture:
    timed: list[tuple[float, str]] = []
    terminal: RuntimeResult | None = None
    start = time.perf_counter()
    for event in runtime.stream_text_turn(
        transcript, min_sentence_chars=24, first_sentence_min_chars=8
    ):
        if event.type == "token":
            timed.append((time.perf_counter() - start, event.text))
        elif event.type == "result" and event.result is not None:
            terminal = event.result
            break
    if terminal is None:
        raise RuntimeError("controlled capture ended without a terminal result")
    return Capture(tuple(timed), terminal)


def _first_flush(
    timed: Sequence[tuple[float, str]], *, first_clause_enabled: bool
) -> FlushMeasurement:
    """Replay tokens and report wall-clock time to the first guarded chunk."""
    runtime = AssistantRuntime(
        llm=_ReplayLLM(list(timed)),
        options=RuntimeOptions(turn_workflow="controlled"),
    )
    start = time.perf_counter()
    first_sentence_ms: float | None = None
    first_sentence = ""
    terminal: RuntimeResult | None = None
    for event in runtime.stream_text_turn(
        "replay",
        min_sentence_chars=24,
        first_sentence_min_chars=8,
        first_clause_enabled=first_clause_enabled,
        first_clause_min_chars=12,
        first_clause_min_words=2,
        first_clause_max_scan_chars=80,
    ):
        if event.type == "sentence" and first_sentence_ms is None:
            first_sentence_ms = (time.perf_counter() - start) * 1000
            first_sentence = event.text
        elif event.type == "result" and event.result is not None:
            terminal = event.result
    if first_sentence_ms is None or terminal is None:
        raise RuntimeError("controlled replay did not produce a complete streamed result")
    return FlushMeasurement(
        elapsed_ms=first_sentence_ms,
        text=first_sentence,
        terminal_route=terminal.route.value,
    )


def _nearest_rank(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = math.ceil(quantile * len(ordered)) - 1
    return ordered[max(0, min(rank, len(ordered) - 1))]


def summarize_measurements(
    values: Sequence[float],
    *,
    helped_threshold_ms: float,
) -> dict[str, int | float | None]:
    return {
        "count": len(values),
        "p50_ms": statistics.median(values) if values else None,
        "p95_ms": _nearest_rank(values, 0.95),
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
        "helped_threshold_ms": helped_threshold_ms,
        "helped_count": sum(value > helped_threshold_ms for value in values),
    }


def run_benchmark() -> dict[str, Any]:
    print(f"loading LLM ({MODEL_ID})...")
    llm = LocalLlamaCppLLM(model_key=MODEL_ID, n_threads=8)
    runtime = AssistantRuntime(
        llm=llm,
        options=RuntimeOptions(turn_workflow="controlled"),
    )
    engine = create_tts_engine(voice=VOICE_ID)

    flush_deltas: list[float] = []
    ready_deltas: list[float] = []
    cases: list[dict[str, Any]] = []
    print()
    header = f"{'transcript':<42} {'flush_off':>9} {'flush_on':>9} {'Δflush':>8} {'Δready':>8}"
    print(header)
    print("-" * len(header))
    for transcript in TRANSCRIPTS:
        capture = _capture(runtime, transcript)
        if len(capture.timed_tokens) < 2:
            print(f"{transcript[:40]:<42} (no LLM tokens - non-chat route, skipped)")
            cases.append(
                {
                    "transcript_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
                    "status": "blocked",
                    "reason": "insufficient_stream_tokens",
                    "token_count": len(capture.timed_tokens),
                    "capture_route": capture.terminal.route.value,
                }
            )
            continue
        on = _first_flush(capture.timed_tokens, first_clause_enabled=True)
        off = _first_flush(capture.timed_tokens, first_clause_enabled=False)

        # real Valtec synth latency of each first chunk
        syn_on_ms = engine.synthesize(on.text).latency_ms if on.text else 0.0
        syn_off_ms = engine.synthesize(off.text).latency_ms if off.text else 0.0
        ready_on_ms = on.elapsed_ms + syn_on_ms
        ready_off_ms = off.elapsed_ms + syn_off_ms

        d_flush = off.elapsed_ms - on.elapsed_ms
        d_ready = ready_off_ms - ready_on_ms
        flush_deltas.append(d_flush)
        ready_deltas.append(d_ready)
        same = "  (same chunk)" if on.text == off.text else ""
        print(
            f"{transcript[:40]:<42} {off.elapsed_ms:>8.0f}m {on.elapsed_ms:>8.0f}m "
            f"{d_flush:>7.0f}m {d_ready:>7.0f}m{same}"
        )
        print(f"    off -> {off.text!r}")
        print(f"    on  -> {on.text!r}")
        cases.append(
            {
                "transcript_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
                "status": "measured",
                "token_count": len(capture.timed_tokens),
                "capture_route": capture.terminal.route.value,
                "on_route": on.terminal_route,
                "off_route": off.terminal_route,
                "flush_on_ms": on.elapsed_ms,
                "flush_off_ms": off.elapsed_ms,
                "tts_on_ms": syn_on_ms,
                "tts_off_ms": syn_off_ms,
                "flush_delta_ms": d_flush,
                "tts_ready_delta_ms": d_ready,
            }
        )

    print("\n===== FIRST-CLAUSE TTFA A/B (positive Δ = first-clause is faster) =====")
    flush_summary = summarize_measurements(flush_deltas, helped_threshold_ms=5.0)
    ready_summary = summarize_measurements(ready_deltas, helped_threshold_ms=5.0)
    print(json.dumps({"flush": flush_summary, "tts_ready": ready_summary}, indent=2))
    return {
        "schema_version": "soca-first-clause-controlled-v2",
        "artifact": make_eval_artifact_metadata(
            suite="first_clause_ttfa_controlled",
            run_type="benchmark",
            data_files=(Path(__file__),),
            config={
                "turn_workflow": "controlled",
                "model": MODEL_ID,
                "voice": VOICE_ID,
                "capture_replay": True,
                "transcript_count": len(TRANSCRIPTS),
                "first_clause_min_chars": 12,
                "first_clause_min_words": 2,
                "first_clause_max_scan_chars": 80,
            },
        ).to_dict(),
        "cases": cases,
        "summary": {"flush_delta": flush_summary, "tts_ready_delta": ready_summary},
        "gate": {
            "passed": len(flush_deltas) == len(TRANSCRIPTS),
            "reason": None if len(flush_deltas) == len(TRANSCRIPTS) else "incomplete_cases",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_benchmark()
    write_json_artifact(args.output, report)
    return int(not report["gate"]["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
