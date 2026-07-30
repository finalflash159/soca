from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, Protocol

DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS = 128


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int:
        ...


class Utf8TokenCounter:
    name = "utf8_bytes_div_4"

    def count(self, text: str) -> int:
        if not text.strip():
            return 0
        return max(1, (len(text.encode("utf-8")) + 3) // 4)


class EngineTokenCounter:
    """Use the active engine tokenizer when it exposes one.

    Provider/local tokenizers are more useful than the UTF-8 estimate, but a
    telemetry helper must never make a turn fail. Invalid tokenizer output or
    a tokenizer exception therefore falls back to the deterministic estimate.
    """

    name = "engine"

    def __init__(self, engine: object) -> None:
        count_tokens = getattr(engine, "count_tokens", None)
        if not callable(count_tokens):
            raise TypeError("engine does not expose count_tokens")
        self._count_tokens = count_tokens
        self._fallback = Utf8TokenCounter()

    def count(self, text: str) -> int:
        try:
            value = self._count_tokens(text)
        except Exception:  # noqa: BLE001 - tokenizer is a best-effort adapter
            return self._fallback.count(text)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return self._fallback.count(text)
        return value


@dataclass(frozen=True)
class ModelCapability:
    model_id: str
    context_window: int | None
    max_output_tokens: int | None = None
    tokenizer: str = "utf8_bytes_div_4"
    source: str = "registry"

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        for name, value in (
            ("context_window", self.context_window),
            ("max_output_tokens", self.max_output_tokens),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer or null")


@dataclass(frozen=True)
class PromptComponent:
    component_id: str
    text: str
    priority: int
    required: bool = False

    def __post_init__(self) -> None:
        if not self.component_id.strip():
            raise ValueError("prompt component id must not be empty")
        if self.priority < 0:
            raise ValueError("prompt component priority must be non-negative")


@dataclass(frozen=True)
class PromptComponentUsage:
    component_id: str
    tokens: int
    included: bool
    required: bool
    priority: int


@dataclass(frozen=True)
class PromptManifest:
    model_id: str
    context_window: int | None
    capability_source: str
    tokenizer: str
    token_counter: str
    requested_output_tokens: int
    effective_output_tokens: int
    input_budget_tokens: int | None
    prompt_tokens: int
    safety_margin_tokens: int
    prompt_hash: str
    components: tuple[PromptComponentUsage, ...]

    @property
    def dropped_components(self) -> tuple[str, ...]:
        return tuple(item.component_id for item in self.components if not item.included)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["components"] = [asdict(item) for item in self.components]
        payload["dropped_components"] = list(self.dropped_components)
        return payload


class PromptBudgetError(ValueError):
    def __init__(self, code: str, *, detail: str = "") -> None:
        super().__init__(detail or code)
        self.code = code
        self.detail = detail


class PromptAssembler:
    def __init__(
        self,
        capability: ModelCapability,
        *,
        counter: TokenCounter | None = None,
        safety_margin_tokens: int = DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS,
        minimum_input_tokens: int = 128,
    ) -> None:
        if safety_margin_tokens < 0 or minimum_input_tokens < 1:
            raise ValueError("prompt budget safety values are invalid")
        self.capability = capability
        self.counter = counter or Utf8TokenCounter()
        self.safety_margin_tokens = safety_margin_tokens
        self.minimum_input_tokens = minimum_input_tokens

    def assemble(
        self,
        components: Iterable[PromptComponent],
        *,
        requested_output_tokens: int,
    ) -> tuple[str, PromptManifest]:
        if requested_output_tokens < 1:
            raise ValueError("requested_output_tokens must be positive")
        original = tuple(components)
        if not original:
            raise PromptBudgetError("empty_prompt")
        component_ids = [item.component_id for item in original]
        if len(component_ids) != len(set(component_ids)):
            raise PromptBudgetError("duplicate_component_id")
        usages = [
            PromptComponentUsage(
                component_id=item.component_id,
                tokens=self.counter.count(item.text),
                included=False,
                required=item.required,
                priority=item.priority,
            )
            for item in original
        ]
        required_tokens = self.counter.count(
            "\n\n".join(item.text for item in original if item.required)
        )
        output_tokens, input_budget = self._budgets(
            requested_output_tokens,
            required_tokens=required_tokens,
        )
        ordered = sorted(enumerate(original), key=lambda item: (item[1].priority, item[0]))
        included: set[str] = set()
        for _, item in ordered:
            if not item.required:
                continue
            candidate = included | {item.component_id}
            if input_budget is not None and self._selected_tokens(original, candidate) > input_budget:
                raise PromptBudgetError(
                    "required_context_overflow",
                    detail=f"required component {item.component_id} exceeds input budget",
                )
            included.add(item.component_id)
        for _, item in ordered:
            if item.required:
                continue
            candidate = included | {item.component_id}
            if input_budget is not None and self._selected_tokens(original, candidate) > input_budget:
                continue
            included.add(item.component_id)

        for index, item in enumerate(original):
            usages[index] = PromptComponentUsage(
                component_id=item.component_id,
                tokens=usages[index].tokens,
                included=item.component_id in included,
                required=item.required,
                priority=item.priority,
            )
        selected = tuple(item.text for item in original if item.component_id in included)
        prompt = "\n\n".join(selected)
        manifest = PromptManifest(
            model_id=self.capability.model_id,
            context_window=self.capability.context_window,
            capability_source=self.capability.source,
            tokenizer=self.capability.tokenizer,
            token_counter=self.counter.name,
            requested_output_tokens=requested_output_tokens,
            effective_output_tokens=output_tokens,
            input_budget_tokens=input_budget,
            prompt_tokens=self.counter.count(prompt),
            safety_margin_tokens=self.safety_margin_tokens,
            prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            components=tuple(usages),
        )
        return prompt, manifest

    def _selected_tokens(
        self,
        components: tuple[PromptComponent, ...],
        included: set[str],
    ) -> int:
        return self.counter.count(
            "\n\n".join(item.text for item in components if item.component_id in included)
        )

    def _budgets(
        self,
        requested_output_tokens: int,
        *,
        required_tokens: int,
    ) -> tuple[int, int | None]:
        max_output = self.capability.max_output_tokens
        effective = min(requested_output_tokens, max_output or requested_output_tokens)
        context_window = self.capability.context_window
        if context_window is None:
            return effective, None
        available_after_required = (
            context_window - self.safety_margin_tokens - required_tokens
        )
        if available_after_required < 1:
            raise PromptBudgetError(
                "required_context_overflow",
                detail="required component set exceeds model context window",
            )
        max_reserve = max(
            1,
            context_window - self.safety_margin_tokens - self.minimum_input_tokens,
        )
        effective = min(effective, max_reserve, available_after_required)
        input_budget = context_window - self.safety_margin_tokens - effective
        return effective, input_budget


def capability_from_engine(
    engine: object | None,
    *,
    model_context_window: int | None = None,
    model_max_output_tokens: int | None = None,
) -> ModelCapability:
    config = getattr(engine, "config", None)
    model_id = str(getattr(engine, "model_key", "") or getattr(engine, "model", "") or "unknown")
    context_window = (
        model_context_window
        if model_context_window is not None
        else getattr(config, "context_window", None)
    )
    if context_window is None:
        context_window = getattr(engine, "n_ctx", None)
    max_output = (
        model_max_output_tokens
        if model_max_output_tokens is not None
        else getattr(engine, "model_max_output_tokens", None)
    )
    tokenizer = str(getattr(engine, "tokenizer_name", "") or "")
    source = "runtime"
    if model_context_window is not None or model_max_output_tokens is not None:
        source = "runtime_options"
    elif config is not None and context_window is not None:
        source = "local_registry"
        tokenizer = tokenizer or "llama_cpp"
    elif model_id != "unknown":
        source = "engine_metadata"
    tokenizer = tokenizer or "utf8_bytes_div_4"
    return ModelCapability(
        model_id=model_id,
        context_window=context_window if isinstance(context_window, int) else None,
        max_output_tokens=max_output,
        tokenizer=tokenizer,
        source=source,
    )


def capability_from_values(
    *,
    model_id: str,
    context_window: int | None,
    max_output_tokens: int | None,
    tokenizer: str = "utf8_bytes_div_4",
    source: str = "settings",
) -> ModelCapability:
    """Build a capability contract for a surface without an active engine."""
    return ModelCapability(
        model_id=model_id.strip() or "unknown",
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        tokenizer=tokenizer,
        source=source,
    )


def token_counter_from_engine(engine: object | None) -> TokenCounter | None:
    if engine is None or not callable(getattr(engine, "count_tokens", None)):
        return None
    return EngineTokenCounter(engine)


__all__ = [
    "ModelCapability",
    "DEFAULT_CONTEXT_SAFETY_MARGIN_TOKENS",
    "PromptAssembler",
    "PromptBudgetError",
    "PromptComponent",
    "PromptComponentUsage",
    "PromptManifest",
    "TokenCounter",
    "EngineTokenCounter",
    "Utf8TokenCounter",
    "capability_from_engine",
    "capability_from_values",
    "token_counter_from_engine",
]
