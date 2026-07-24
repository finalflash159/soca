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

import statistics
import time
from collections.abc import Iterator

from soca.core import AssistantRuntime
from soca.llm import LocalLlamaCppLLM
from soca.llm.base import LLMResult
from soca.tts import create_tts_engine

TRANSCRIPTS = [
    "Chào bạn, bạn là ai vậy?",
    "Giúp tôi kiểm tra lại cấu hình một chút nhé.",
    "Bạn có thể tóm tắt ngắn gọn giúp tôi không?",
    "Hôm nay tôi nên ăn gì để đủ chất đạm?",
    "Kể cho tôi nghe một chút về bản thân bạn đi.",
    "Bạn thấy trời hôm nay thế nào?",
    "Cho tôi một lời khuyên để ngủ ngon hơn nhé.",
    "Giải thích giúp tôi vì sao nên uống đủ nước.",
]


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


def _capture(runtime: AssistantRuntime, transcript: str) -> list[tuple[float, str]]:
    timed: list[tuple[float, str]] = []
    start = time.perf_counter()
    for event in runtime.stream_text_turn(
        transcript, min_sentence_chars=24, first_sentence_min_chars=8
    ):
        if event.type == "token":
            timed.append((time.perf_counter() - start, event.text))
        elif event.type == "result":
            break
    return timed


def _first_flush(
    timed: list[tuple[float, str]], *, first_clause_enabled: bool
) -> tuple[float, str]:
    """Replay tokens; return (wall-clock seconds to first sentence, first chunk text)."""
    runtime = AssistantRuntime(llm=_ReplayLLM(timed))
    start = time.perf_counter()
    for event in runtime.stream_text_turn(
        "replay",
        min_sentence_chars=24,
        first_sentence_min_chars=8,
        first_clause_enabled=first_clause_enabled,
        first_clause_min_chars=12,
        first_clause_min_words=2,
        first_clause_max_scan_chars=80,
    ):
        if event.type == "sentence":
            return time.perf_counter() - start, event.text
    return float("nan"), ""


def main() -> int:
    print("loading LLM (arcee_vylinh_3b_q4_k_m)...")
    llm = LocalLlamaCppLLM(model_key="arcee_vylinh_3b_q4_k_m", n_threads=8)
    runtime = AssistantRuntime(llm=llm)
    engine = create_tts_engine(voice="NF")

    flush_deltas: list[float] = []
    ready_deltas: list[float] = []
    print()
    header = f"{'transcript':<42} {'flush_off':>9} {'flush_on':>9} {'Δflush':>8} {'Δready':>8}"
    print(header)
    print("-" * len(header))
    for transcript in TRANSCRIPTS:
        timed = _capture(runtime, transcript)
        if len(timed) < 2:
            print(f"{transcript[:40]:<42} (no LLM tokens - non-chat route, skipped)")
            continue
        t_on, chunk_on = _first_flush(timed, first_clause_enabled=True)
        t_off, chunk_off = _first_flush(timed, first_clause_enabled=False)

        # real Valtec synth latency of each first chunk
        syn_on = engine.synthesize(chunk_on).latency_ms / 1000.0 if chunk_on else 0.0
        syn_off = engine.synthesize(chunk_off).latency_ms / 1000.0 if chunk_off else 0.0
        ready_on = t_on + syn_on
        ready_off = t_off + syn_off

        d_flush = (t_off - t_on) * 1000.0
        d_ready = (ready_off - ready_on) * 1000.0
        flush_deltas.append(d_flush)
        ready_deltas.append(d_ready)
        same = "  (same chunk)" if chunk_on == chunk_off else ""
        print(
            f"{transcript[:40]:<42} {t_off*1000:>8.0f}m {t_on*1000:>8.0f}m "
            f"{d_flush:>7.0f}m {d_ready:>7.0f}m{same}"
        )
        print(f"    off -> {chunk_off!r}")
        print(f"    on  -> {chunk_on!r}")

    def _summ(name: str, values: list[float]) -> None:
        if not values:
            print(f"{name}: (none)")
            return
        print(
            f"{name}: p50={statistics.median(values):+.0f}ms "
            f"min={min(values):+.0f}ms max={max(values):+.0f}ms (n={len(values)})"
        )

    print("\n===== FIRST-CLAUSE TTFA A/B (positive Δ = first-clause is faster) =====")
    _summ("Δ time-to-first-sentence (text side)", flush_deltas)
    _summ("Δ tts_ready (text + Valtec synth)   ", ready_deltas)
    helped = sum(1 for d in flush_deltas if d > 5)
    print(f"prompts where first-clause flushed earlier (>5ms): {helped}/{len(flush_deltas)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
