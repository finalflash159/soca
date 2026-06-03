from __future__ import annotations

import queue
import random
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from soca.core.audio_out import AudioSink, NullAudioPlayer
from soca.core.metrics import MetricsLogger
from soca.core.repair import (
    RepairAction,
    RepairCatalog,
    RepairChoice,
    RepairKind,
    RepairState,
    plan_repair,
)
from soca.core.streaming import StreamingEvent, pop_ready_sentence
from soca.core.text_chunking import chunk_text_for_tts, normalize_text_for_tts
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
    # Repair layer (plan §9): user-facing follow-up for a rejected turn. Kept
    # alongside the legacy rejected/rejection_reason fields for back-compat.
    repair_kind: str = ""
    repair_action: str = ""
    repair_attempt: int = 0
    handover_target: str | None = None


class VoicePipeline:
    def __init__(
        self,
        asr: Any,
        llm: Any,
        tts: Any,
        assistant_runtime: Any | None = None,
        metrics: MetricsLogger | None = None,
        reject_response: str = "[laughter] Moshi moshi? Có ai đó hăm?",
        repair_catalog: RepairCatalog | None = None,
    ) -> None:
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.assistant_runtime = assistant_runtime
        self.metrics = metrics or MetricsLogger()
        # Deterministic single-string fallback (used when no catalog is wired,
        # e.g. unit tests). Production injects ``repair_catalog`` for variety.
        self.reject_response = reject_response
        self.repair_catalog = repair_catalog
        self._repair_state = RepairState()
        self._repair_rng = random.Random()

    def _plan_repair(self, rejection_reason: str) -> RepairChoice:
        """Plan the user-facing follow-up for an empty/rejected ASR turn.

        Picks a fresh natural variant from the catalog (with escalation +
        no-repeat) when available; otherwise wraps the fixed ``reject_response``
        for backward compatibility.
        """
        if self.repair_catalog is None:
            return RepairChoice(
                text=self.reject_response,
                prompt_id="legacy.reject_response",
                kind=RepairKind.NO_INPUT,
                action=RepairAction.REPROMPT,
            )
        return plan_repair(
            self.repair_catalog,
            rejection_reason=rejection_reason,
            state=self._repair_state,
            rng=self._repair_rng,
        )

    @staticmethod
    def _repair_metadata(choice: RepairChoice, *, attempt: int) -> dict[str, Any]:
        handover = "chat" if choice.action == RepairAction.HANDOVER_TO_CHAT else None
        return {
            "repair_kind": choice.kind.value,
            "repair_action": choice.action.value,
            "repair_attempt": attempt,
            "handover_target": handover,
        }

    def turn(self, audio: np.ndarray) -> PipelineResult:
        self.metrics.reset()
        t0 = time.perf_counter()

        with self.metrics.stage("asr"):
            asr_result = self.asr.transcribe(audio)

        transcript = getattr(asr_result, "text", "").strip()
        rejection_reason = getattr(asr_result, "rejection_reason", "")

        if not transcript:
            repair = self._plan_repair(rejection_reason)
            return PipelineResult(
                transcript="",
                response_text=repair.text,
                rejected=True,
                rejection_reason=rejection_reason or "empty_transcript",
                tts=None,
                stage_latencies_ms=self.metrics.snapshot(),
                total_latency_ms=(time.perf_counter() - t0) * 1000,
                asr_result=asr_result,
                repair_kind=repair.kind.value,
                repair_action=repair.action.value,
                repair_attempt=self._repair_state.no_input_attempts,
                handover_target="chat" if repair.action == RepairAction.HANDOVER_TO_CHAT else None,
            )

        self._repair_state.reset()  # successful turn clears the repair ladder
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
            tts_result = self.tts.synthesize(_speech_text(response_text))

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
        first_sentence_min_chars: int = 8,
        speak_rejections: bool = True,
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
            repair = self._plan_repair(rejection_reason)
            repair_meta = self._repair_metadata(
                repair, attempt=self._repair_state.no_input_attempts
            )
            # Repair event lands before sentence/tts so UI can label the turn as a
            # follow-up (not an error) and act on handover (plan §9).
            yield StreamingEvent(
                type="repair",
                text=repair.text,
                metadata={**repair_meta, "technical_reason": rejection_reason or "empty_transcript"},
            )
            if speak_rejections:
                chunks = chunk_text_for_tts(repair.text, min_chars=min_sentence_chars)
                for sentence in chunks:
                    yield StreamingEvent(type="sentence", text=sentence)
                yield from self._stream_tts_playback(
                    chunks,
                    sink=sink,
                    turn_start_time=t0,
                )

            yield StreamingEvent(
                type="done",
                text=repair.text,
                latency_ms=(time.perf_counter() - t0) * 1000,
                metadata={
                    "rejected": True,
                    "rejection_reason": rejection_reason or "empty_transcript",
                    "stage_latencies_ms": self.metrics.snapshot(),
                    **repair_meta,
                },
            )
            return

        self._repair_state.reset()  # successful turn clears the repair ladder
        if self.assistant_runtime is not None:
            if hasattr(self.assistant_runtime, "stream_text_turn"):
                yield from self._turn_streaming_runtime_stream(
                    transcript,
                    rejection_reason,
                    turn_start_time=t0,
                    sink=sink,
                    min_sentence_chars=min_sentence_chars,
                    first_sentence_min_chars=first_sentence_min_chars,
                )
            else:
                yield from self._turn_streaming_runtime_blocking(
                    transcript,
                    rejection_reason,
                    turn_start_time=t0,
                    sink=sink,
                    min_sentence_chars=min_sentence_chars,
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

                    speech_text = _speech_text(sentence)
                    with self.metrics.stage(f"tts_{index}"):
                        tts_result = self.tts.synthesize(speech_text)

                    if first_audio_time is None:
                        first_audio_time = time.perf_counter()

                    metadata: dict[str, float | int] = {"chunk_index": index}
                    if index == 0:
                        metadata["ttfa_ms"] = (first_audio_time - t0) * 1000

                    event_queue.put(
                        StreamingEvent(
                            type="tts",
                            text=speech_text,
                            audio=tts_result.audio,
                            sample_rate=tts_result.sample_rate,
                            tts=tts_result,
                            latency_ms=tts_result.latency_ms,
                            metadata=metadata,
                        )
                    )

                    playback = sink.play(tts_result.audio, tts_result.sample_rate, blocking=True)

                    audio_metadata = dict(metadata)
                    audio_metadata["playback_latency_ms"] = playback.latency_ms
                    event_queue.put(
                        StreamingEvent(
                            type="audio",
                            text=speech_text,
                            audio=tts_result.audio,
                            sample_rate=tts_result.sample_rate,
                            tts=tts_result,
                            latency_ms=playback.latency_ms,
                            metadata=audio_metadata,
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

        worker = threading.Thread(target=tts_worker, name="soca-tts-worker")
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

    def _runtime_summary_event(self, runtime_result: Any) -> StreamingEvent:
        """Build the ``runtime`` summary event from a completed RuntimeResult."""
        trace = getattr(runtime_result, "trace", None)
        citations = getattr(runtime_result, "citations", ())
        return StreamingEvent(
            type="runtime",
            text=getattr(runtime_result, "response_text", "").strip(),
            metadata={
                "route": getattr(getattr(runtime_result, "route", None), "value", ""),
                "blocked": bool(getattr(runtime_result, "blocked", False)),
                "used_tool": bool(getattr(trace, "used_tool", False)),
                "used_llm": bool(getattr(trace, "used_llm", False)),
                "citations": [
                    {"path": item.path, "title": item.title} for item in citations
                ],
                # LLM telemetry for `soca voice --usage`. Object is fine in metadata;
                # eval/console read named keys, not the whole dict.
                "llm_usage": getattr(runtime_result, "usage", None),
            },
        )

    def _turn_streaming_runtime_blocking(
        self,
        transcript: str,
        rejection_reason: str,
        *,
        turn_start_time: float,
        sink: AudioSink,
        min_sentence_chars: int,
    ) -> Iterator[StreamingEvent]:
        """Legacy path for runtimes without a streaming API.

        The whole response is generated before TTS starts; TTS synthesis and
        playback are still pipelined across sentences.
        """
        with self.metrics.stage("runtime"):
            runtime_result = self.assistant_runtime.run_text_turn(
                transcript,
                source="asr",
                metadata={"asr_rejection_reason": rejection_reason},
            )

        response_text = getattr(runtime_result, "response_text", "").strip()
        yield self._runtime_summary_event(runtime_result)

        chunks = chunk_text_for_tts(response_text, min_chars=min_sentence_chars)
        for sentence in chunks:
            yield StreamingEvent(type="sentence", text=sentence)

        yield from self._stream_tts_playback(
            chunks,
            sink=sink,
            turn_start_time=turn_start_time,
        )

        yield StreamingEvent(
            type="done",
            text=response_text,
            latency_ms=(time.perf_counter() - turn_start_time) * 1000,
            metadata={
                "rejected": False,
                "runtime_blocked": bool(getattr(runtime_result, "blocked", False)),
                "runtime_route": getattr(getattr(runtime_result, "route", None), "value", ""),
                "stage_latencies_ms": self.metrics.snapshot(),
            },
        )

    def _turn_streaming_runtime_stream(
        self,
        transcript: str,
        rejection_reason: str,
        *,
        turn_start_time: float,
        sink: AudioSink,
        min_sentence_chars: int,
        first_sentence_min_chars: int | None = None,
    ) -> Iterator[StreamingEvent]:
        """True end-to-end streaming: LLM tokens feed TTS as sentences complete.

        The runtime yields tokens and guardrail-passed sentences incrementally.
        Each sentence is queued to a TTS worker, whose audio is queued to a
        playback worker, so synthesis of sentence N+1 overlaps playback of
        sentence N. The first sentence reaches the speaker without waiting for
        the full response to be generated.
        """
        sentence_queue: queue.Queue[str | None] = queue.Queue()
        playback_queue: queue.Queue[StreamingEvent | object] = queue.Queue()
        event_queue: queue.Queue[StreamingEvent | object] = queue.Queue()
        done = object()

        def tts_worker() -> None:
            first_audio_time: float | None = None
            index = 0
            try:
                while True:
                    sentence = sentence_queue.get()
                    if sentence is None:
                        return

                    speech_text = _speech_text(sentence)
                    with self.metrics.stage(f"tts_{index}"):
                        tts_result = self.tts.synthesize(speech_text)

                    if first_audio_time is None:
                        first_audio_time = time.perf_counter()

                    metadata: dict[str, float | int] = {
                        "chunk_index": index,
                        "tts_latency_ms": tts_result.latency_ms,
                    }
                    if index == 0:
                        metadata["ttfa_ms"] = (first_audio_time - turn_start_time) * 1000

                    event = StreamingEvent(
                        type="tts",
                        text=speech_text,
                        audio=tts_result.audio,
                        sample_rate=tts_result.sample_rate,
                        tts=tts_result,
                        latency_ms=tts_result.latency_ms,
                        metadata=metadata,
                    )
                    event_queue.put(event)
                    playback_queue.put(event)
                    index += 1
            except Exception as exc:
                event_queue.put(StreamingEvent(type="error", text=str(exc)))
            finally:
                playback_queue.put(done)
                event_queue.put(done)

        def playback_worker() -> None:
            try:
                while True:
                    event = playback_queue.get()
                    if event is done:
                        return

                    assert isinstance(event, StreamingEvent)
                    assert event.audio is not None
                    assert event.sample_rate is not None
                    playback = sink.play(event.audio, event.sample_rate, blocking=True)
                    metadata = dict(event.metadata or {})
                    metadata["playback_latency_ms"] = playback.latency_ms
                    event_queue.put(
                        StreamingEvent(
                            type="audio",
                            text=event.text,
                            audio=event.audio,
                            sample_rate=event.sample_rate,
                            tts=event.tts,
                            latency_ms=playback.latency_ms,
                            metadata=metadata,
                        )
                    )
            except Exception as exc:
                event_queue.put(StreamingEvent(type="error", text=str(exc)))
            finally:
                event_queue.put(done)

        def drain_ready() -> Iterator[StreamingEvent]:
            while True:
                try:
                    event = event_queue.get_nowait()
                except queue.Empty:
                    return
                if event is done:
                    event_queue.put(done)
                    return
                assert isinstance(event, StreamingEvent)
                yield event

        tts_thread = threading.Thread(target=tts_worker, name="soca-runtime-stream-tts")
        playback_thread = threading.Thread(
            target=playback_worker,
            name="soca-runtime-stream-playback",
        )
        tts_thread.start()
        playback_thread.start()

        runtime_result: Any | None = None
        for event in self.assistant_runtime.stream_text_turn(
            transcript,
            source="asr",
            metadata={"asr_rejection_reason": rejection_reason},
            min_sentence_chars=min_sentence_chars,
            first_sentence_min_chars=first_sentence_min_chars,
        ):
            if event.type == "token":
                yield StreamingEvent(type="llm_token", text=event.text)
            elif event.type == "sentence":
                yield StreamingEvent(type="sentence", text=event.text)
                sentence_queue.put(event.text)
            elif event.type == "result":
                runtime_result = event.result
            yield from drain_ready()

        sentence_queue.put(None)  # Signal TTS worker that no more sentences arrive.

        if runtime_result is not None:
            yield self._runtime_summary_event(runtime_result)

        completed_workers = 0
        while completed_workers < 2:
            event = event_queue.get()
            if event is done:
                completed_workers += 1
                continue
            assert isinstance(event, StreamingEvent)
            yield event

        tts_thread.join()
        playback_thread.join()

        yield StreamingEvent(
            type="done",
            text=getattr(runtime_result, "response_text", "").strip(),
            latency_ms=(time.perf_counter() - turn_start_time) * 1000,
            metadata={
                "rejected": False,
                "runtime_blocked": bool(getattr(runtime_result, "blocked", False)),
                "runtime_route": getattr(getattr(runtime_result, "route", None), "value", ""),
                "stage_latencies_ms": self.metrics.snapshot(),
            },
        )

    def _stream_tts_playback(
        self,
        sentences: list[str],
        *,
        sink: AudioSink,
        turn_start_time: float,
    ) -> Iterator[StreamingEvent]:
        event_queue: queue.Queue[StreamingEvent | object] = queue.Queue()
        playback_queue: queue.Queue[StreamingEvent | object] = queue.Queue()
        done = object()

        def tts_worker() -> None:
            first_audio_time: float | None = None

            try:
                for index, sentence in enumerate(sentences):
                    speech_text = _speech_text(sentence)
                    with self.metrics.stage(f"tts_{index}"):
                        tts_result = self.tts.synthesize(speech_text)

                    audio_ready_time = time.perf_counter()
                    if first_audio_time is None:
                        first_audio_time = audio_ready_time

                    metadata: dict[str, float | int] = {
                        "chunk_index": index,
                        "tts_latency_ms": tts_result.latency_ms,
                    }
                    if index == 0:
                        metadata["ttfa_ms"] = (first_audio_time - turn_start_time) * 1000

                    event = StreamingEvent(
                        type="tts",
                        text=speech_text,
                        audio=tts_result.audio,
                        sample_rate=tts_result.sample_rate,
                        tts=tts_result,
                        latency_ms=tts_result.latency_ms,
                        metadata=metadata,
                    )
                    event_queue.put(event)
                    playback_queue.put(event)
            except Exception as exc:
                event_queue.put(StreamingEvent(type="error", text=str(exc)))
            finally:
                playback_queue.put(done)
                event_queue.put(done)

        def playback_worker() -> None:
            try:
                while True:
                    event = playback_queue.get()
                    if event is done:
                        return

                    assert isinstance(event, StreamingEvent)
                    assert event.audio is not None
                    assert event.sample_rate is not None
                    playback = sink.play(event.audio, event.sample_rate, blocking=True)
                    metadata = dict(event.metadata or {})
                    metadata["playback_latency_ms"] = playback.latency_ms
                    event_queue.put(
                        StreamingEvent(
                            type="audio",
                            text=event.text,
                            audio=event.audio,
                            sample_rate=event.sample_rate,
                            tts=event.tts,
                            latency_ms=playback.latency_ms,
                            metadata=metadata,
                        )
                    )
            except Exception as exc:
                event_queue.put(StreamingEvent(type="error", text=str(exc)))
            finally:
                event_queue.put(done)

        tts_thread = threading.Thread(target=tts_worker, name="soca-runtime-tts-worker")
        playback_thread = threading.Thread(
            target=playback_worker,
            name="soca-runtime-playback-worker",
        )
        tts_thread.start()
        playback_thread.start()

        completed_workers = 0
        while completed_workers < 2:
            event = event_queue.get()
            if event is done:
                completed_workers += 1
                continue

            assert isinstance(event, StreamingEvent)
            yield event

        tts_thread.join()
        playback_thread.join()


def _speech_text(text: str) -> str:
    return normalize_text_for_tts(text) or text.strip()
