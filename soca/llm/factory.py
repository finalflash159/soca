"""Select the local or remote LLM engine from persisted non-secret settings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from soca.config.llm_settings import LlmSettings
from soca.llm.base import LLMEngine
from soca.llm.providers import RemoteLLMError, RemoteOpenAILLM, get_provider


class SecretReader(Protocol):
    def get_key(self, provider_key: str) -> str | None: ...


LocalFactory = Callable[..., LLMEngine]
RemoteFactory = Callable[..., LLMEngine]


def build_llm_engine(
    settings: LlmSettings,
    secrets: SecretReader,
    *,
    local_factory: LocalFactory | None = None,
    remote_factory: RemoteFactory = RemoteOpenAILLM,
    n_threads: int = 8,
    n_gpu_layers: int = -1,
) -> LLMEngine:
    """Build an LLM without ever exposing the resolved key to callers."""
    if settings.backend == "local":
        factory = local_factory or _local_llama_factory
        return factory(
            model_key=settings.model_id,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
        )

    provider = get_provider(settings.provider_key)
    api_key = secrets.get_key(provider.key)
    if not api_key:
        raise RemoteLLMError(
            f"Chưa có API key cho {provider.label}. Hãy nhập key trước khi chọn provider này.",
            category="auth",
        )
    return remote_factory(provider, settings.model_id, api_key)


def _local_llama_factory(
    *,
    model_key: str,
    n_threads: int,
    n_gpu_layers: int,
) -> LLMEngine:
    from soca.llm import LocalLlamaCppLLM

    return LocalLlamaCppLLM(
        model_key=model_key,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
    )


__all__ = ["SecretReader", "build_llm_engine"]
