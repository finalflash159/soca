from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from soca.core.audio_out import AudioSink, NullAudioPlayer
from soca.core.metrics import MetricsLogger
from soca.core.streaming import StreamingEvent, pop_ready_sentence
from soca.core.text_chunking import split_sentences
from soca.tts import TTSResult


@dataclass(frozen=True)
class PipelineResult:
    transcript: str
    response_text: str
    rejected: bool
    rejection_reason: str
    tts: TTSResult | None
    stage_latencies_ms: dict[str, float]
    total_latency_ms: float
    asr_result: Any | None = None
    llm_result: Any | None = None
    runtime_result: Any | None = None


class VoicePipeline:
    def __init__(
        self,
        asr: Any,
        llm: Any,
        tts: Any,
        assistant_runtime: Any | None = None,
        metrics: MetricsLogger | None = None,
        reject_response: str = "Mình chưa nghe rõ, bạn nói lại nhé.",
    ) -> None:
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.assistant_runtime = assistant_runtime
        self.metrics = metrics or MetricsLogger()
        self.reject_response = reject_response

    def turn(self, audio: np.ndarray) -> PipelineResult:
        self.metrics.reset()
        t0 = time.perf_counter()

        with self.metrics.stage("asr"):
            asr_result = self.asr.transcribe(audio)

        transcript = getattr(asr_result, "text", "").strip()
        rejection_reason = getattr(asr_result, "rejection_reason", "")

        if not transcript:
            return PipelineResult(
                transcript="",
                response_text=self.reject_response,
                rejected=True,
                rejection_reason=rejection_reason or "empty_transcript",
                tts=None,
                stage_latencies_ms=self.metrics.snapshot(),
                total_latency_ms=(time.perf_counter() - t0) * 1000,
                asr_result=asr_result,
            )

        llm_result = None
        runtime_result = None
        if self.assistant_runtime is not None:
            with self.metrics.stage("runtime"):
                runtime_result = self.assistant_runtime.run_text_turn(
                    transcript,
                    source="asr",
                    metadata={
                        "asr_rejection_reason": rejection_reason,
                    },
                )
            response_text = getattr(runtime_result, "response_text", "").strip()
            llm_result = getattr(runtime_result, "llm_result", None)
        else:
            with self.metrics.stage("llm"):
                llm_result = self.llm.generate(transcript)

            response_text = getattr(llm_result, "text", "").strip()

        with self.metrics.stage("tts"):
            tts_result = self.tts.synthesize(response_text)

        return PipelineResult(
            transcript=transcript,
            response_text=response_text,
            rejected=False,
            rejection_reason="",
            tts=tts_result,
            stage_latencies_ms=self.metrics.snapshot(),
            total_latency_ms=(time.perf_counter() - t0) * 1000,
            asr_result=asr_result,
            llm_result=llm_result,
            runtime_result=runtime_result,
        )

    def turn_streaming(
        self,
        audio: np.ndarray,
        audio_sink: AudioSink | None = None,
        min_sentence_chars: int = 24,
    ) -> Iterator[StreamingEvent]:
        self.metrics.reset()
        t0 = time.perf_counter()
        sink = audio_sink or NullAudioPlayer()

        with self.metrics.stage("asr"):
            asr_result = self.asr.transcribe(audio)

        transcript = getattr(asr_result, "text", "").strip()
        rejection_reason = getattr(asr_result, "rejection_reason", "")

        yield StreamingEvent(
            type="asr",
            text=transcript,
            metadata={"rejection_reason": rejection_reason},
        )

        if not transcript:
            yield StreamingEvent(
                type="done",
                text=self.reject_response,
                latency_ms=(time.perf_counter() - t0) * 1000,
                metadata={
                    "rejected": True,
                    "rejection_reason": rejection_reason or "empty_transcript",
                },
            )
            return

        if self.assistant_runtime is not None:
            with self.metrics.stage("runtime"):
                runtime_result = self.assistant_runtime.run_text_turn(
                    transcript,
                    source="asr",
                    metadata={
                        "asr_rejection_reason": rejection_reason,
                    },
                )

            response_text = getattr(runtime_result, "response_text", "").strip()
            trace = getattr(runtime_result, "trace", None)
            citations = getattr(runtime_result, "citations", ())
            yield StreamingEvent(
                type="runtime",
                text=response_text,
                metadata={
                    "route": getattr(getattr(runtime_result, "route", None), "value", ""),
                    "blocked": bool(getattr(runtime_result, "blocked", False)),
                    "used_tool": bool(getattr(trace, "used_tool", False)),
                    "used_llm": bool(getattr(trace, "used_llm", False)),
                    "citations": [
                        {"path": item.path, "title": item.title}
                        for item in citations
                    ],
                },
            )

            first_audio_time: float | None = None
            for index, sentence in enumerate(split_sentences(response_text)):
                yield StreamingEvent(type="sentence", text=sentence)

                with self.metrics.stage(f"tts_{index}"):
                    tts_result = self.tts.synthesize(sentence)

                if first_audio_time is None:
                    first_audio_time = time.perf_counter()

                yield StreamingEvent(
                    type="tts",
                    text=sentence,
                    audio=tts_result.audio,
                    sample_rate=tts_result.sample_rate,
                    tts=tts_result,
                    latency_ms=tts_result.latency_ms,
                    metadata={
                        "chunk_index": index,
                        "ttfa_ms": (first_audio_time - t0) * 1000,
                    },
                )

                playback = sink.play(tts_result.audio, tts_result.sample_rate, blocking=True)

                yield StreamingEvent(
                    type="audio",
                    text=sentence,
                    audio=tts_result.audio,
                    sample_rate=tts_result.sample_rate,
                    tts=tts_result,
                    latency_ms=playback.latency_ms,
                    metadata={
                        "chunk_index": index,
                        "ttfa_ms": (first_audio_time - t0) * 1000,
                    },
                )

            yield StreamingEvent(
                type="done",
                text=response_text,
                latency_ms=(time.perf_counter() - t0) * 1000,
                metadata={
                    "rejected": False,
                    "runtime_blocked": bool(getattr(runtime_result, "blocked", False)),
                    "runtime_route": getattr(getattr(runtime_result, "route", None), "value", ""),
                    "stage_latencies_ms": self.metrics.snapshot(),
                },
            )
            return

        sentence_queue: queue.Queue[str | None] = queue.Queue()
        event_queue: queue.Queue[StreamingEvent | None] = queue.Queue()

        def drain_ready_events() -> Iterator[StreamingEvent]:
            while True:
                try:
                    event = event_queue.get_nowait()
                except queue.Empty:
                    return
                if event is not None:
                    yield event
                else:
                    event_queue.put(None)
                    return

        def tts_worker() -> None:
            first_audio_time: float | None = None
            index = 0

            try:
                while True:
                    sentence = sentence_queue.get()
                    if sentence is None:
                        event_queue.put(None)
                        return

                    with self.metrics.stage(f"tts_{index}"):
                        tts_result = self.tts.synthesize(sentence)

                    if first_audio_time is None:
                        first_audio_time = time.perf_counter()

                    event_queue.put(
                        StreamingEvent(
                            type="tts",
                            text=sentence,
                            audio=tts_result.audio,
                            sample_rate=tts_result.sample_rate,
                            tts=tts_result,
                            latency_ms=tts_result.latency_ms,
                            metadata={
                                "chunk_index": index,
                                "ttfa_ms": (first_audio_time - t0) * 1000,
                            },
                        )
                    )

                    playback = sink.play(tts_result.audio, tts_result.sample_rate, blocking=True)

                    event_queue.put(
                        StreamingEvent(
                            type="audio",
                            text=sentence,
                            audio=tts_result.audio,
                            sample_rate=tts_result.sample_rate,
                            tts=tts_result,
                            latency_ms=playback.latency_ms,
                            metadata={
                                "chunk_index": index,
                                "ttfa_ms": (first_audio_time - t0) * 1000,
                            },
                        )
                    )
                    index += 1
            except Exception as exc:
                event_queue.put(
                    StreamingEvent(
                        type="error",
                        text=str(exc),
                        metadata={"chunk_index": index},
                    )
                )
                event_queue.put(None)

                return

        worker = threading.Thread(target=tts_worker, daemon=True)
        worker.start()

        buffer = ""
        response_parts: list[str] = []

        with self.metrics.stage("llm"):
            for token in self.llm.generate_stream(transcript):
                response_parts.append(token)
                buffer += token

                yield StreamingEvent(type="llm_token", text=token)
                yield from drain_ready_events()

                while True:
                    sentence, buffer = pop_ready_sentence(buffer, min_chars=min_sentence_chars)
                    if sentence is None:
                        break

                    yield StreamingEvent(type="sentence", text=sentence)
                    sentence_queue.put(sentence)
                    yield from drain_ready_events()

        final_text = buffer.strip()
        if final_text:
            sentence_queue.put(final_text)
            yield StreamingEvent(type="sentence", text=final_text)

        sentence_queue.put(None)  # Signal TTS worker to exit

        while True:
            event = event_queue.get()
            if event is None:
                break
            yield event

        worker.join()

        yield StreamingEvent(
            type="done",
            text="".join(response_parts).strip(),
            latency_ms=(time.perf_counter() - t0) * 1000,
            metadata={
                "rejected": False,
                "stage_latencies_ms": self.metrics.snapshot(),
            },
        )
