from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from soca.asr.qwen_artifacts import (
    QWEN_ARTIFACT_REGISTRY,
    ArtifactRole,
)
from soca.asr.registry import ASR_MODEL_REGISTRY


class ASREngine(StrEnum):
    PHOWHISPER = "phowhisper"
    QWEN_SERVICE = "qwen_service"


class ASRSelectionError(ValueError):
    """An ASR engine, model and role combination is invalid."""


@dataclass(frozen=True, slots=True)
class ASRSelection:
    engine: ASREngine
    model_key: str
    artifact_role: ArtifactRole | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.engine, ASREngine):
            raise ASRSelectionError("ASR engine must be typed")
        if not isinstance(self.model_key, str) or not self.model_key:
            raise ASRSelectionError("ASR model key must be a non-empty string")

        if self.engine is ASREngine.PHOWHISPER:
            self._validate_phowhisper()
            return
        self._validate_qwen_service()

    def _validate_phowhisper(self) -> None:
        if self.model_key not in ASR_MODEL_REGISTRY:
            valid = ", ".join(sorted(ASR_MODEL_REGISTRY))
            raise ASRSelectionError(
                f"unknown PhoWhisper model key: {self.model_key}; valid keys: {valid}"
            )
        if self.artifact_role is not None:
            raise ASRSelectionError("PhoWhisper selection must not declare a Qwen artifact role")

    def _validate_qwen_service(self) -> None:
        try:
            artifact = QWEN_ARTIFACT_REGISTRY[self.model_key]
        except KeyError as exc:
            valid = ", ".join(sorted(QWEN_ARTIFACT_REGISTRY))
            raise ASRSelectionError(
                f"unknown Qwen ASR artifact key: {self.model_key}; valid keys: {valid}"
            ) from exc
        if not isinstance(self.artifact_role, ArtifactRole):
            raise ASRSelectionError("Qwen service selection requires a typed artifact role")
        if artifact.role is not self.artifact_role:
            raise ASRSelectionError(
                f"Qwen artifact {self.model_key} has role {artifact.role.value}, "
                f"not {self.artifact_role.value}"
            )

    @classmethod
    def phowhisper(cls, model_key: str) -> ASRSelection:
        return cls(engine=ASREngine.PHOWHISPER, model_key=model_key)

    @classmethod
    def qwen_service(
        cls,
        artifact_key: str,
        *,
        role: ArtifactRole,
    ) -> ASRSelection:
        return cls(
            engine=ASREngine.QWEN_SERVICE,
            model_key=artifact_key,
            artifact_role=role,
        )


__all__ = ["ASREngine", "ASRSelection", "ASRSelectionError"]
