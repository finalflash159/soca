from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from soca.core.guardrails import (
    DEFAULT_POLICY,
    GuardrailEvent,
    GuardrailPolicy,
    GuardrailStage,
    check_final_output,
    check_input_text,
    check_tool_call,
    check_tool_result,
    check_untrusted_text,
    extract_markdown_paths,
    is_time_question,
    normalize_vi,
)
from soca.core.streaming import pop_ready_first_clause, pop_ready_sentence
from soca.core.text_chunking import chunk_text_for_tts
from soca.core.turn import (
    RuntimeResult,
    RuntimeRoute,
    RuntimeStreamEvent,
    RuntimeTrace,
    TurnFrame,
)
from soca.core.usage import LLMUsage
from soca.knowledge import KnowledgeCitation, KnowledgeContext, KnowledgeContextBuilder
from soca.llm import LLMEngine
from soca.memory import MemoryContext, MemoryContextBuilder
from soca.prompts import build_runtime_prompt
from soca.tools import ToolCall, ToolResult, ToolRuntime


@dataclass(frozen=True)
class RuntimeOptions:
    max_tokens: int = 128
    temperature: float = 0.2
    top_p: float = 0.95
    knowledge_limit: int = 3


class RuntimeToolRouter(Protocol):
    def select(self, text: str, *, knowledge_limit: int) -> ToolCall | None: ...


class DefaultRuntimeToolRouter:
    """Small deterministic router for explicit local capabilities.

    This router intentionally avoids project/domain keywords. Knowledge retrieval
    is explicit command syntax by default; richer natural-language routing should
    be added as a separate policy/router, not baked into AssistantRuntime.
    """

    def __init__(
        self,
        *,
        knowledge_search_prefixes: tuple[str, ...] = ("wiki:", "knowledge:"),
        enable_markdown_read: bool = True,
        enable_time: bool = True,
    ) -> None:
        self.knowledge_search_prefixes = knowledge_search_prefixes
        self.enable_markdown_read = enable_markdown_read
        self.enable_time = enable_time

    def select(self, text: str, *, knowledge_limit: int) -> ToolCall | None:
        if self.enable_markdown_read:
            path = self._first_markdown_path(text)
            if path is not None:
                return ToolCall("knowledge.read", {"path": path})

        if self.enable_time and is_time_question(text):
            return ToolCall("local_time.now", {})

        query = self._parse_knowledge_search_query(text)
        if query is not None:
            return ToolCall(
                "knowledge.search",
                {
                    "query": query,
                    "limit": knowledge_limit,
                },
            )

        return None

    def _first_markdown_path(self, text: str) -> str | None:
        paths = extract_markdown_paths(text)
        return paths[0] if paths else None

    def _parse_knowledge_search_query(self, text: str) -> str | None:
        stripped = text.strip()
        normalized = normalize_vi(stripped)
        for prefix in self.knowledge_search_prefixes:
            normalized_prefix = normalize_vi(prefix)
            if normalized.startswith(normalized_prefix):
                query = stripped[len(prefix) :].strip(" :,-")
                return query or None
        return None


@dataclass(frozen=True)
class _TraceDraft:
    guardrail_events: list[GuardrailEvent]
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    knowledge_hits: list[Any]
    citations: list[KnowledgeCitation]
    stage_latencies_ms: dict[str, float]


class AssistantRuntime:
    """Deterministic text-turn runtime between ASR and TTS."""

    def __init__(
        self,
        llm: LLMEngine | None = None,
        *,
        tool_runtime: ToolRuntime | None = None,
        tool_router: RuntimeToolRouter | None = None,
        knowledge_builder: KnowledgeContextBuilder | None = None,
        memory_builder: MemoryContextBuilder | None = None,
        guardrail_policy: GuardrailPolicy = DEFAULT_POLICY,
        options: RuntimeOptions | None = None,
    ) -> None:
        self.llm = llm
        self.tool_runtime = tool_runtime or ToolRuntime()
        self.tool_router = tool_router or DefaultRuntimeToolRouter()
        self.knowledge_builder = knowledge_builder
        self.memory_builder = memory_builder
        self.guardrail_policy = guardrail_policy
        self.options = options or RuntimeOptions()

    def run_text_turn(
        self,
        text: str,
        *,
        source: str = "text",
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeResult:
        frame = TurnFrame(text=text, source=source, metadata=metadata or {})
        draft = _TraceDraft([], [], [], [], [], {})

        with self._stage(draft, "input_guardrail"):
            input_event = check_input_text(frame.text, self.guardrail_policy)
        draft.guardrail_events.append(input_event)
        if input_event.blocked:
            return self._blocked_result(
                frame,
                draft,
                reason=input_event.message or self._safe_block_message(input_event.reason),
            )

        tool_call = self.tool_router.select(
            frame.text,
            knowledge_limit=self.options.knowledge_limit,
        )
        if tool_call is not None:
            return self._run_tool_turn(frame, tool_call, draft)

        memory_context = self._build_memory_context(draft)
        knowledge_context = self._build_knowledge_context(frame, draft)

        return self._run_llm_turn(frame, draft, memory_context, knowledge_context)

    def stream_text_turn(
        self,
        text: str,
        *,
        source: str = "text",
        metadata: dict[str, Any] | None = None,
        min_sentence_chars: int = 24,
        first_sentence_min_chars: int | None = None,
        first_clause_enabled: bool = True,
        first_clause_min_chars: int = 12,
        first_clause_min_words: int = 2,
        first_clause_max_scan_chars: int = 80,
    ) -> Iterator[RuntimeStreamEvent]:
        """Streaming counterpart of run_text_turn.

        Yields ``token`` events for each raw LLM token, ``sentence`` events for
        each guardrail-passed chunk ready for TTS, and a final ``result`` event
        carrying the complete RuntimeResult. Only the LLM route streams
        token-by-token; tool/knowledge-direct and blocked routes resolve to a
        fixed text that is chunked and emitted as sentences.

        ``first_sentence_min_chars`` lets the very first chunk flush at a smaller
        length than the rest, so the first audio reaches the speaker sooner
        (lower time-to-first-audio). When ``None`` every chunk uses
        ``min_sentence_chars``.
        """
        frame = TurnFrame(text=text, source=source, metadata=metadata or {})
        draft = _TraceDraft([], [], [], [], [], {})

        with self._stage(draft, "input_guardrail"):
            input_event = check_input_text(frame.text, self.guardrail_policy)
        draft.guardrail_events.append(input_event)
        if input_event.blocked:
            result = self._blocked_result(
                frame,
                draft,
                reason=input_event.message or self._safe_block_message(input_event.reason),
            )
            yield from self._emit_fixed_result(result, min_sentence_chars=min_sentence_chars)
            return

        tool_call = self.tool_router.select(
            frame.text,
            knowledge_limit=self.options.knowledge_limit,
        )
        if tool_call is not None:
            result = self._run_tool_turn(frame, tool_call, draft)
            yield from self._emit_fixed_result(result, min_sentence_chars=min_sentence_chars)
            return

        memory_context = self._build_memory_context(draft)
        knowledge_context = self._build_knowledge_context(frame, draft)
        yield from self._stream_llm_turn(
            frame,
            draft,
            memory_context,
            knowledge_context,
            min_sentence_chars=min_sentence_chars,
            first_sentence_min_chars=first_sentence_min_chars,
            first_clause_enabled=first_clause_enabled,
            first_clause_min_chars=first_clause_min_chars,
            first_clause_min_words=first_clause_min_words,
            first_clause_max_scan_chars=first_clause_max_scan_chars,
        )

    def _emit_fixed_result(
        self,
        result: RuntimeResult,
        *,
        min_sentence_chars: int,
    ) -> Iterator[RuntimeStreamEvent]:
        """Emit a non-streamed result: chunk its text, then the result event."""
        for chunk in chunk_text_for_tts(result.response_text, min_chars=min_sentence_chars):
            yield RuntimeStreamEvent(type="sentence", text=chunk)
        yield RuntimeStreamEvent(type="result", result=result)

    def _guard_sentence(
        self,
        sentence: str,
        knowledge_used: bool,
        citations: tuple[KnowledgeCitation, ...],
    ) -> GuardrailEvent:
        """Apply the streaming-safe subset of the output guardrail to one chunk.

        For the LLM route, check_final_output reduces to a stateless substring
        scan (realtime-claim detection), which decomposes cleanly per sentence,
        so this is semantically equivalent to checking the full response.
        """
        return check_final_output(
            sentence,
            knowledge_used=knowledge_used,
            citations=citations,
            tool_results=(),
            realtime_tool_used=False,
            policy=self.guardrail_policy,
        )

    def _build_stream_usage(
        self,
        prompt: str,
        completion: str,
        *,
        started: float,
        first_token_time: float | None,
        ended: float,
    ) -> LLMUsage:
        """Normalized LLM telemetry for the streaming route.

        Mirrors LocalLlamaCppLLM.generate(): TTFT from the first token, tok/s over
        the decode window. Token counts use the engine's tokenizer when available
        (``count_tokens``), falling back to a whitespace approximation for fakes
        or engines that don't expose one.
        """
        count_tokens = getattr(self.llm, "count_tokens", None)
        if callable(count_tokens):
            prompt_tokens = count_tokens(prompt)
            completion_tokens = count_tokens(completion)
        else:
            prompt_tokens = 0
            completion_tokens = len(completion.split())

        total_latency_ms = (ended - started) * 1000
        if first_token_time is None:
            return LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                ttft_ms=total_latency_ms,
                total_latency_ms=total_latency_ms,
                tokens_per_second=0.0,
            )

        ttft_ms = (first_token_time - started) * 1000
        gen_time = ended - first_token_time
        tps = (max(completion_tokens - 1, 0) / gen_time) if gen_time > 0 else 0.0
        return LLMUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            tokens_per_second=tps,
        )

    def _stream_llm_turn(
        self,
        frame: TurnFrame,
        draft: _TraceDraft,
        memory_context: MemoryContext | None,
        knowledge_context: KnowledgeContext | None,
        *,
        min_sentence_chars: int,
        first_sentence_min_chars: int | None = None,
        first_clause_enabled: bool = True,
        first_clause_min_chars: int = 12,
        first_clause_min_words: int = 2,
        first_clause_max_scan_chars: int = 80,
    ) -> Iterator[RuntimeStreamEvent]:
        if self.llm is None:
            result = self._blocked_result(
                frame,
                draft,
                reason="Mình chưa có LLM để trả lời câu này.",
                route=RuntimeRoute.BLOCKED,
            )
            yield from self._emit_fixed_result(result, min_sentence_chars=min_sentence_chars)
            return

        prompt = self._build_llm_prompt(frame.text, memory_context, knowledge_context)
        citations = tuple(draft.citations)
        knowledge_used = bool(citations)

        buffer = ""
        response_parts: list[str] = []
        spoken_sentences: list[str] = []
        block_event: GuardrailEvent | None = None

        started = time.perf_counter()
        first_token_time: float | None = None
        stream = self.llm.generate_stream(
            prompt,
            max_tokens=self.options.max_tokens,
            temperature=self.options.temperature,
            top_p=self.options.top_p,
            inject_persona=False,
        )
        try:
            for token in stream:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                response_parts.append(token)
                buffer += token
                yield RuntimeStreamEvent(type="token", text=token)

                while True:
                    sentence: str | None = None
                    if first_clause_enabled and not spoken_sentences:
                        sentence, next_buffer = pop_ready_first_clause(
                            buffer,
                            min_chars=first_clause_min_chars,
                            min_words=first_clause_min_words,
                            max_scan_chars=first_clause_max_scan_chars,
                        )
                        if sentence is not None:
                            buffer = next_buffer

                    if sentence is None:
                        active_min = (
                            first_sentence_min_chars
                            if (not spoken_sentences and first_sentence_min_chars is not None)
                            else min_sentence_chars
                        )
                        sentence, buffer = pop_ready_sentence(
                            buffer,
                            min_chars=active_min,
                        )

                    if sentence is None:
                        break

                    guard = self._guard_sentence(
                        sentence,
                        knowledge_used,
                        citations,
                    )
                    draft.guardrail_events.append(guard)
                    if guard.blocked:
                        block_event = guard
                        break
                    spoken_sentences.append(sentence)
                    yield RuntimeStreamEvent(type="sentence", text=sentence)
                if block_event is not None:
                    break
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        ended = time.perf_counter()
        draft.stage_latencies_ms["llm"] = (ended - started) * 1000

        if block_event is None:
            tail = buffer.strip()
            if tail:
                guard = self._guard_sentence(tail, knowledge_used, citations)
                draft.guardrail_events.append(guard)
                if guard.blocked:
                    block_event = guard
                else:
                    spoken_sentences.append(tail)
                    yield RuntimeStreamEvent(type="sentence", text=tail)

        full_text = "".join(response_parts).strip()
        usage = self._build_stream_usage(
            prompt,
            full_text,
            started=started,
            first_token_time=first_token_time,
            ended=ended,
        )

        if block_event is None:
            # Belt-and-suspenders: re-check the full text in case a blocked
            # phrase straddled two streamed sentence boundaries. Per-sentence
            # checks above cover the common case, so this rarely fires.
            with self._stage(draft, "output_guardrail"):
                final_event = check_final_output(
                    full_text,
                    knowledge_used=knowledge_used,
                    citations=citations,
                    tool_results=tuple(draft.tool_results),
                    realtime_tool_used=False,
                    policy=self.guardrail_policy,
                )
            draft.guardrail_events.append(final_event)
            if final_event.blocked:
                block_event = final_event

        if block_event is not None:
            safe_message = block_event.message or self._safe_block_message(block_event.reason)
            yield RuntimeStreamEvent(type="sentence", text=safe_message)
            spoken = " ".join([*spoken_sentences, safe_message]).strip()
            result = self._result(
                frame,
                draft,
                response_text=spoken,
                route=RuntimeRoute.BLOCKED,
                used_tool=False,
                used_llm=True,
                blocked=True,
                usage=usage,
            )
            yield RuntimeStreamEvent(type="result", result=result)
            return

        self._append_safe_session_turn(frame.text, full_text)
        result = self._result(
            frame,
            draft,
            response_text=full_text,
            route=RuntimeRoute.KNOWLEDGE_LLM if knowledge_used else RuntimeRoute.FREE_CHAT,
            used_tool=False,
            used_llm=True,
            usage=usage,
        )
        yield RuntimeStreamEvent(type="result", result=result)

    def _run_tool_turn(
        self,
        frame: TurnFrame,
        tool_call: ToolCall,
        draft: _TraceDraft,
    ) -> RuntimeResult:
        draft.tool_calls.append(tool_call)

        with self._stage(draft, "tool_input_guardrail"):
            tool_input_event = check_tool_call(
                self.tool_runtime,
                tool_call,
                self.guardrail_policy,
            )
        draft.guardrail_events.append(tool_input_event)
        if tool_input_event.blocked:
            return self._blocked_result(
                frame,
                draft,
                reason=tool_input_event.message
                or self._safe_block_message(tool_input_event.reason),
                route=RuntimeRoute.BLOCKED,
            )

        with self._stage(draft, f"tool:{tool_call.name}"):
            tool_result = self.tool_runtime.call(tool_call)
        draft.tool_results.append(tool_result)

        with self._stage(draft, "tool_output_guardrail"):
            tool_output_event = check_tool_result(tool_result)
        draft.guardrail_events.append(tool_output_event)
        if tool_output_event.blocked:
            return self._blocked_result(
                frame,
                draft,
                reason=tool_output_event.message
                or self._safe_block_message(tool_output_event.reason),
                route=RuntimeRoute.BLOCKED,
            )

        citations = self._citations_from_tool_result(tool_result)
        draft.citations.extend(citations)
        route = (
            RuntimeRoute.KNOWLEDGE_DIRECT
            if tool_call.name.startswith("knowledge.")
            else RuntimeRoute.TOOL_DIRECT
        )

        knowledge_used = bool(citations)
        with self._stage(draft, "output_guardrail"):
            output_event = check_final_output(
                tool_result.content,
                knowledge_used=knowledge_used,
                citations=tuple(citations),
                tool_results=(tool_result,),
                realtime_tool_used=tool_call.name == "local_time.now",
                policy=self.guardrail_policy,
            )
        draft.guardrail_events.append(output_event)
        if output_event.blocked:
            return self._blocked_result(
                frame,
                draft,
                reason=output_event.message or self._safe_block_message(output_event.reason),
                route=RuntimeRoute.BLOCKED,
            )

        response_text = tool_result.content.strip()
        self._append_safe_session_turn(frame.text, response_text)
        return self._result(
            frame,
            draft,
            response_text=response_text,
            route=route,
            used_tool=True,
            used_llm=False,
        )

    def _run_llm_turn(
        self,
        frame: TurnFrame,
        draft: _TraceDraft,
        memory_context: MemoryContext | None,
        knowledge_context: KnowledgeContext | None,
    ) -> RuntimeResult:
        if self.llm is None:
            return self._blocked_result(
                frame,
                draft,
                reason="Mình chưa có LLM để trả lời câu này.",
                route=RuntimeRoute.BLOCKED,
            )

        prompt = self._build_llm_prompt(frame.text, memory_context, knowledge_context)

        with self._stage(draft, "llm"):
            llm_result = self.llm.generate(
                prompt,
                max_tokens=self.options.max_tokens,
                temperature=self.options.temperature,
                top_p=self.options.top_p,
                inject_persona=False,
            )

        response_text = getattr(llm_result, "text", "").strip()
        usage = LLMUsage.from_llm_result(llm_result)
        citations = tuple(draft.citations)
        knowledge_used = bool(citations)

        with self._stage(draft, "output_guardrail"):
            output_event = check_final_output(
                response_text,
                knowledge_used=knowledge_used,
                citations=citations,
                tool_results=tuple(draft.tool_results),
                realtime_tool_used=False,
                policy=self.guardrail_policy,
            )
        draft.guardrail_events.append(output_event)
        if output_event.blocked:
            return self._blocked_result(
                frame,
                draft,
                reason=output_event.message or self._safe_block_message(output_event.reason),
                route=RuntimeRoute.BLOCKED,
                llm_result=llm_result,
                usage=usage,
            )

        self._append_safe_session_turn(frame.text, response_text)
        return self._result(
            frame,
            draft,
            response_text=response_text,
            route=RuntimeRoute.KNOWLEDGE_LLM if knowledge_used else RuntimeRoute.FREE_CHAT,
            used_tool=False,
            used_llm=True,
            llm_result=llm_result,
            usage=usage,
        )

    def _build_memory_context(self, draft: _TraceDraft) -> MemoryContext | None:
        if self.memory_builder is None:
            return None

        with self._stage(draft, "memory_context"):
            context = self.memory_builder.build()
        return context

    def _build_knowledge_context(
        self,
        frame: TurnFrame,
        draft: _TraceDraft,
    ) -> KnowledgeContext | None:
        if self.knowledge_builder is None:
            return None
        if not self._should_build_knowledge_context(frame):
            return None

        query = str(frame.metadata.get("knowledge_query") or frame.text)
        with self._stage(draft, "knowledge_context"):
            context = self.knowledge_builder.build(query)

        draft.knowledge_hits.extend(context.hits)
        draft.citations.extend(context.citations)
        for hit in context.hits:
            draft.guardrail_events.append(
                check_untrusted_text(hit.snippet, stage=GuardrailStage.RETRIEVAL)
            )
        return context

    def _build_llm_prompt(
        self,
        user_text: str,
        memory_context: MemoryContext | None,
        knowledge_context: KnowledgeContext | None,
    ) -> str:
        return build_runtime_prompt(
            user_text=user_text,
            memory_prompt_text=memory_context.prompt_text if memory_context is not None else "",
            knowledge_prompt_text=(
                knowledge_context.prompt_text if knowledge_context is not None else ""
            ),
        )

    def _append_safe_session_turn(self, user_text: str, assistant_text: str) -> None:
        if self.memory_builder is None or self.memory_builder.session is None:
            return
        self.memory_builder.session.append("user", user_text)
        self.memory_builder.session.append("assistant", assistant_text)

    def _result(
        self,
        frame: TurnFrame,
        draft: _TraceDraft,
        *,
        response_text: str,
        route: RuntimeRoute,
        used_tool: bool,
        used_llm: bool,
        blocked: bool = False,
        llm_result: Any | None = None,
        usage: LLMUsage | None = None,
    ) -> RuntimeResult:
        trace = RuntimeTrace(
            route=route,
            guardrail_events=tuple(draft.guardrail_events),
            tool_calls=tuple(draft.tool_calls),
            tool_results=tuple(draft.tool_results),
            knowledge_hits=tuple(draft.knowledge_hits),
            citations=tuple(draft.citations),
            used_tool=used_tool,
            used_llm=used_llm,
            blocked=blocked,
            stage_latencies_ms=dict(draft.stage_latencies_ms),
        )
        return RuntimeResult(
            response_text=response_text,
            route=route,
            blocked=blocked,
            citations=trace.citations,
            trace=trace,
            frame=frame,
            llm_result=llm_result,
            usage=usage,
        )

    def _blocked_result(
        self,
        frame: TurnFrame,
        draft: _TraceDraft,
        *,
        reason: str,
        route: RuntimeRoute = RuntimeRoute.BLOCKED,
        llm_result: Any | None = None,
        usage: LLMUsage | None = None,
    ) -> RuntimeResult:
        return self._result(
            frame,
            draft,
            response_text=reason,
            route=route,
            used_tool=bool(draft.tool_calls),
            used_llm=llm_result is not None or usage is not None,
            blocked=True,
            llm_result=llm_result,
            usage=usage,
        )

    def _safe_block_message(self, reason: str) -> str:
        if reason == "unknown_tool":
            return "Mình chưa có công cụ phù hợp để làm việc đó."
        if reason == "invalid_tool_arguments":
            return "Mình chưa hiểu đủ tham số để gọi công cụ."
        if reason == "side_effect_not_allowed":
            return "Công cụ đó vượt quá quyền runtime hiện tại."
        return "Mình không thể xử lý yêu cầu này một cách an toàn."

    def _should_build_knowledge_context(self, frame: TurnFrame) -> bool:
        return bool(frame.metadata.get("use_knowledge") or frame.metadata.get("knowledge_query"))

    def _citations_from_tool_result(self, result: ToolResult) -> tuple[KnowledgeCitation, ...]:
        if not result.name.startswith("knowledge."):
            return ()

        if "hits" in result.data:
            citations: list[KnowledgeCitation] = []
            for hit in result.data.get("hits", []):
                path = str(hit.get("path", ""))
                title = str(hit.get("title", path))
                if path:
                    citations.append(KnowledgeCitation(path=path, title=title))
            return tuple(citations)

        path = str(result.data.get("path", ""))
        title = str(result.data.get("title", path))
        if not path:
            return ()
        return (KnowledgeCitation(path=path, title=title),)

    @contextmanager
    def _stage(self, draft: _TraceDraft, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            draft.stage_latencies_ms[name] = (time.perf_counter() - started) * 1000
