"""Real-device playback measurement for Phase 7 (pump -> session -> SoundDevicePlayer).

Bypasses ASR/LLM (they produce no device metric); TTS->pump->speaker is real.
Plays Valtec audio through the default output device and reports the device-only
telemetry the offline A/B eval cannot produce.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from soca.core import (
    RuntimeResult,
    RuntimeRoute,
    RuntimeStreamEvent,
    RuntimeTrace,
    SoundDevicePlayer,
    VoicePipeline,
)
from soca.tts import create_tts_engine


@dataclass(frozen=True)
class _ASRResult:
    text: str
    rejection_reason: str = ""


class _FakeASR:
    def __init__(self, text: str) -> None:
        self._text = text

    def transcribe(self, audio: np.ndarray) -> _ASRResult:
        del audio
        return _ASRResult(text=self._text)


class _SentenceRuntime:
    """Yields fixed sentences like a streaming LLM would (no model needed)."""

    def __init__(self, sentences: list[str]) -> None:
        self.sentences = sentences

    def stream_text_turn(
        self,
        text: str,
        *,
        source: str = "text",
        metadata=None,
        min_sentence_chars: int = 24,
        first_sentence_min_chars: int | None = None,
        first_clause_enabled: bool = True,
        first_clause_min_chars: int = 12,
        first_clause_min_words: int = 2,
        first_clause_max_scan_chars: int = 80,
    ) -> Iterator[RuntimeStreamEvent]:
        del (
            text,
            source,
            metadata,
            min_sentence_chars,
            first_sentence_min_chars,
            first_clause_enabled,
            first_clause_min_chars,
            first_clause_min_words,
            first_clause_max_scan_chars,
        )
        for sentence in self.sentences:
            for token in sentence.split(" "):
                yield RuntimeStreamEvent(type="token", text=token + " ")
            yield RuntimeStreamEvent(type="sentence", text=sentence)
        yield RuntimeStreamEvent(
            type="result",
            result=RuntimeResult(
                response_text=" ".join(self.sentences),
                route=RuntimeRoute.FREE_CHAT,
                blocked=False,
                trace=RuntimeTrace(
                    route=RuntimeRoute.FREE_CHAT,
                    used_tool=False,
                    used_llm=True,
                    blocked=False,
                ),
            ),
        )


RESPONSES = [
    [
        "Mình đã nhận yêu cầu của bạn rồi nhé.",
        "Bây giờ mình sẽ bắt đầu xử lý và báo lại kết quả ngay khi xong.",
    ],
    [
        "Tuy nhiên, mình cần kiểm tra thêm một chút.",
        "Sau khi xác nhận dữ liệu đầu vào, mình sẽ chạy đánh giá thật.",
        "Kết quả chi tiết sẽ được lưu vào báo cáo cho bạn xem lại.",
    ],
    [
        "Độ trễ đã giảm rõ rệt so với trước.",
        "Chất lượng phát âm vẫn được giữ nguyên, không có tiếng rè ở biên nối.",
    ],
    [
        "Cụm đầu đã sẵn sàng và loa bắt đầu phát ngay.",
        "Cụm thứ hai vẫn đang được tổng hợp song song ở phía sau.",
        "Nhờ vậy bạn không phải chờ hết câu mới nghe được tiếng nói.",
    ],
    [
        "Được rồi, mình chốt cấu hình mặc định ở mười hai mili giây.",
        "Nếu cần, bạn có thể chỉnh xuống tám mili giây trong hồ sơ giọng nói.",
    ],
]


def main() -> int:
    engine = create_tts_engine(voice="NF")
    print("engine role:", engine.artifact_metadata["role"], "| voices:", engine.list_voices())

    audible_ttfa: list[float] = []
    tts_ready_ttfa: list[float] = []
    slacks: list[float] = []
    crossfades: list[float] = []
    total_underflow = 0
    total_fallback = 0
    total_boundaries = 0

    for i, sentences in enumerate(RESPONSES):
        pipeline = VoicePipeline(
            asr=_FakeASR("kiểm tra giúp tôi"),
            llm=object(),
            tts=engine,
            assistant_runtime=_SentenceRuntime(sentences),
        )
        sink = SoundDevicePlayer()  # real device; crossfade default 12ms keeps session path
        events = list(
            pipeline.turn_streaming(
                np.zeros(16_000, dtype=np.float32),
                audio_sink=sink,
                min_sentence_chars=24,
            )
        )
        audio_events = [e for e in events if e.type == "audio"]
        done = next(e for e in events if e.type == "done")
        summary = (done.metadata or {}).get("playback", {})

        for e in audio_events:
            meta = e.metadata or {}
            if "audible_ttfa_ms" in meta:
                audible_ttfa.append(meta["audible_ttfa_ms"])
            if "tts_ready_ttfa_ms" in meta:
                tts_ready_ttfa.append(meta["tts_ready_ttfa_ms"])
            if "synthesis_slack_ms" in meta:
                slacks.append(meta["synthesis_slack_ms"])
                total_boundaries += 1
            if meta.get("crossfade_ms"):
                crossfades.append(meta["crossfade_ms"])
        total_underflow += int(summary.get("output_underflow_count", 0))
        total_fallback += int(summary.get("crossfade_fallback_count", 0))
        print(
            f"turn {i}: chunks={len(audio_events)} "
            f"audible_ttfa={audible_ttfa[-1]:.0f}ms "
            f"underflow={summary.get('output_underflow_count')} "
            f"fallback={summary.get('crossfade_fallback_count')}"
        )

    def _stat(values: list[float], name: str) -> str:
        if not values:
            return f"{name}: (none)"
        p50 = statistics.median(values)
        p05 = float(np.percentile(values, 5))
        p95 = float(np.percentile(values, 95))
        return f"{name}: p05={p05:.1f} p50={p50:.1f} p95={p95:.1f} (n={len(values)})"

    print("\n===== DEVICE PLAYBACK SUMMARY (real speaker) =====")
    print(_stat(tts_ready_ttfa, "tts_ready_ttfa_ms"))
    print(_stat(audible_ttfa, "audible_ttfa_ms  "))
    print(_stat(slacks, "synthesis_slack_ms"))
    print(_stat(crossfades, "crossfade_ms      "))
    print(f"boundaries measured: {total_boundaries}")
    print(f"output_underflow_count TOTAL: {total_underflow}  (gate == 0)")
    fallback_rate = (total_fallback / total_boundaries * 100.0) if total_boundaries else 0.0
    print(f"crossfade_fallback TOTAL: {total_fallback}  ({fallback_rate:.1f}% of boundaries, gate < 1%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
