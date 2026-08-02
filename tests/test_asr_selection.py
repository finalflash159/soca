from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from soca.asr.qwen_artifacts import ArtifactRole
from soca.asr.selection import ASREngine, ASRSelection, ASRSelectionError


def test_phowhisper_selection_validates_its_registry() -> None:
    selection = ASRSelection.phowhisper("phowhisper_small")

    assert selection.engine is ASREngine.PHOWHISPER
    assert selection.model_key == "phowhisper_small"
    assert selection.artifact_role is None


def test_qwen_selection_validates_artifact_and_typed_role() -> None:
    selection = ASRSelection.qwen_service(
        "qwen3_asr_0_6b",
        role=ArtifactRole.RELEASE,
    )

    assert selection.engine is ASREngine.QWEN_SERVICE
    assert selection.model_key == "qwen3_asr_0_6b"
    assert selection.artifact_role is ArtifactRole.RELEASE


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        (
            lambda: ASRSelection.phowhisper("qwen3_asr_0_6b"),
            "unknown PhoWhisper",
        ),
        (
            lambda: ASRSelection(
                engine=ASREngine.QWEN_SERVICE,
                model_key="phowhisper_small",
                artifact_role=ArtifactRole.RELEASE,
            ),
            "unknown Qwen",
        ),
        (
            lambda: ASRSelection(
                engine=ASREngine.QWEN_SERVICE,
                model_key="qwen3_asr_0_6b",
            ),
            "typed artifact role",
        ),
        (
            lambda: ASRSelection.qwen_service(
                "qwen3_asr_0_6b",
                role=ArtifactRole.REFERENCE,
            ),
            "has role release",
        ),
        (
            lambda: ASRSelection(
                engine=ASREngine.PHOWHISPER,
                model_key="phowhisper_small",
                artifact_role=ArtifactRole.RELEASE,
            ),
            "must not declare",
        ),
    ],
)
def test_cross_registry_and_role_mismatches_are_rejected(
    selection: object,
    message: str,
) -> None:
    with pytest.raises(ASRSelectionError, match=message):
        selection()  # type: ignore[operator]


def test_selection_requires_typed_engine_and_is_immutable() -> None:
    with pytest.raises(ASRSelectionError, match="engine must be typed"):
        ASRSelection(engine="phowhisper", model_key="phowhisper_small")  # type: ignore[arg-type]

    selection = ASRSelection.phowhisper("phowhisper_small")
    with pytest.raises(FrozenInstanceError):
        selection.model_key = "phowhisper_base"  # type: ignore[misc]
