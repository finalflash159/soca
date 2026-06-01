"""Generic local LLM runner backed by llama-cpp-python.

Despite the upstream `Llama` class name, this runner is for GGUF models that
llama.cpp supports: PhoGPT/MPT, Qwen, VinaLLaMA, Sailor, Gemma, Mistral, and
other compatible architectures.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from llama_cpp import Llama

from .base import LLMResult
from .prompts import (
    StreamingOutputCleaner,
    build_chat_messages,
    build_completion_prompt,
    clean_model_output,
    uses_completion_prompt,
)
from .registry import DEFAULT_LLM_MODEL_KEY, get_model_config


class LocalLlamaCppLLM:
    def __init__(
        self,
        model_key: str = DEFAULT_LLM_MODEL_KEY,
        model_path: str | Path | None = None,
        n_ctx: int | None = None,
        n_threads: int = 4,
        n_gpu_layers: int = -1,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        self.config = get_model_config(model_key)
        self.model_key = model_key
        self.model_path = Path(model_path or self.config.local_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Run: {self.config.download_command}"
            )

        self.n_ctx = n_ctx or min(self.config.context_window, 4096)

        llama_kwargs: dict[str, Any] = {
            "model_path": str(self.model_path),
            "n_ctx": self.n_ctx,
            "n_threads": n_threads,
            "n_gpu_layers": n_gpu_layers,
            "seed": seed,
            "verbose": verbose,
            "use_mlock": False,
        }
        if self.config.chat_format is not None:
            llama_kwargs["chat_format"] = self.config.chat_format

        self.llm = Llama(**llama_kwargs)

    @staticmethod
    def _validate_generation_args(
        user_msg: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> None:
        if not user_msg.strip():
            raise ValueError("user_msg must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in the range (0, 1]")

    @staticmethod
    def _chunk_text(chunk: dict) -> str:
        choices = chunk.get("choices", [])
        if not choices:
            return ""

        choice = choices[0]
        if "text" in choice:
            return choice.get("text") or ""

        delta = choice.get("delta") or {}
        return delta.get("content") or ""

    def _stream_chunks(
        self,
        user_msg: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        inject_persona: bool,
    ) -> Iterator[dict]:
        if uses_completion_prompt(self.config):
            prompt = build_completion_prompt(user_msg, self.config, inject_persona)
            yield from self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=list(self.config.stop_sequences),
                stream=True,
            )
            return

        messages = build_chat_messages(user_msg, self.config, inject_persona)
        yield from self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
        )

    def _prompt_for_metrics(self, user_msg: str, inject_persona: bool) -> str:
        if uses_completion_prompt(self.config):
            return build_completion_prompt(user_msg, self.config, inject_persona)
        messages = build_chat_messages(user_msg, self.config, inject_persona)
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def generate(
        self,
        user_msg: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        inject_persona: bool = True,
    ) -> LLMResult:
        self._validate_generation_args(user_msg, max_tokens, temperature, top_p)

        t0 = time.perf_counter()
        first_token_time: float | None = None
        chunks: list[str] = []

        for chunk in self._stream_chunks(user_msg, max_tokens, temperature, top_p, inject_persona):
            tok_text = self._chunk_text(chunk)
            if first_token_time is None and tok_text:
                first_token_time = time.perf_counter()
            if tok_text:
                chunks.append(tok_text)

        t1 = time.perf_counter()

        raw_completion = "".join(chunks)
        completion_text = clean_model_output(raw_completion, self.config)
        prompt = self._prompt_for_metrics(user_msg, inject_persona)

        n_prompt = len(self.llm.tokenize(prompt.encode("utf-8")))
        n_completion = len(self.llm.tokenize(raw_completion.encode("utf-8"), add_bos=False))

        total_latency_ms = (t1 - t0) * 1000
        ttft_ms = (first_token_time - t0) * 1000 if first_token_time else total_latency_ms

        gen_time = (t1 - first_token_time) if first_token_time else (t1 - t0)
        tps_tokens = max(n_completion - 1, 0) if first_token_time else n_completion
        tps = tps_tokens / gen_time if gen_time > 0 else 0.0

        return LLMResult(
            text=completion_text,
            prompt=prompt,
            n_prompt_tokens=n_prompt,
            n_completion_tokens=n_completion,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            tokens_per_second=tps,
        )

    def generate_stream(
        self,
        user_msg: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        inject_persona: bool = True,
    ) -> Iterator[str]:
        self._validate_generation_args(user_msg, max_tokens, temperature, top_p)
        cleaner = StreamingOutputCleaner(self.config)

        for chunk in self._stream_chunks(user_msg, max_tokens, temperature, top_p, inject_persona):
            tok_text = self._chunk_text(chunk)
            if tok_text:
                yield from cleaner.feed(tok_text)

        yield from cleaner.flush()


__all__ = ["LocalLlamaCppLLM"]
