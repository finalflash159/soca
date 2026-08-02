"""Select the local or remote LLM engine from persisted non-secret settings."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from soca.config.llm_settings import LlmSettings
from soca.llm.base import LLMEngine
from soca.llm.providers import RemoteLLMError, RemoteOpenAILLM, get_provider
from soca.llm.providers.provider_registry import LLMProvider
from soca.llm.providers.request_adapter import ReasoningParameter


class SecretReader(Protocol):
    def get_key(self, provider_key: str) -> str | None: ...


LocalFactory = Callable[..., LLMEngine]


class RemoteFactory(Protocol):
    def __call__(
        self,
        provider: LLMProvider,
        model: str,
        api_key: str,
        *,
        reasoning_enabled: bool | None,
        reasoning_parameter: ReasoningParameter | None,
        max_output_tokens: int,
    ) -> LLMEngine: ...


class EngineBuilder(Protocol):
    def __call__(
        self,
        settings: LlmSettings,
        secrets: SecretReader,
        *,
        local_factory: LocalFactory | None = None,
        n_threads: int = 8,
        n_gpu_layers: int = -1,
    ) -> LLMEngine: ...


@dataclass(frozen=True)
class LlmEngineFactory:
    remote_factory: RemoteFactory = RemoteOpenAILLM

    def __call__(
        self,
        settings: LlmSettings,
        secrets: SecretReader,
        *,
        local_factory: LocalFactory | None = None,
        n_threads: int = 8,
        n_gpu_layers: int = -1,
    ) -> LLMEngine:
        return _build_llm_engine(
            settings,
            secrets,
            local_factory=local_factory,
            remote_factory=self.remote_factory,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
        )


DEFAULT_LLM_ENGINE_FACTORY = LlmEngineFactory()


def build_llm_engine(
    settings: LlmSettings,
    secrets: SecretReader,
    *,
    local_factory: LocalFactory | None = None,
    remote_factory: RemoteFactory = RemoteOpenAILLM,
    n_threads: int = 8,
    n_gpu_layers: int = -1,
) -> LLMEngine:
    return _build_llm_engine(
        settings,
        secrets,
        local_factory=local_factory,
        remote_factory=remote_factory,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
    )


def _build_llm_engine(
    settings: LlmSettings,
    secrets: SecretReader,
    *,
    local_factory: LocalFactory | None,
    remote_factory: RemoteFactory,
    n_threads: int,
    n_gpu_layers: int,
) -> LLMEngine:
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
    return remote_factory(
        provider,
        settings.model_id,
        api_key,
        reasoning_enabled=settings.effective_reasoning_enabled,
        reasoning_parameter=settings.model_reasoning_parameter,
        max_output_tokens=settings.effective_max_tokens,
    )


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


__all__ = [
    "DEFAULT_LLM_ENGINE_FACTORY",
    "EngineBuilder",
    "LlmEngineFactory",
    "SecretReader",
    "build_llm_engine",
]
