from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from soca.asr.calibration import (
    DEFAULT_VAD_POLICY_DIGEST,
    ASRCalibrationError,
    ASRCalibrationNotReady,
    compute_vad_policy_digest,
    load_strict_confidence_calibration,
    qwen_calibration_identity,
)
from soca.asr.qwen_artifacts import QWEN_RELEASE_ARTIFACT, ArtifactRole
from soca.asr.result import ASRResult
from soca.asr.selection import ASREngine
from soca.asr.vad import VADResult
from soca.core import voice_runtime
from soca.core.voice_runtime import _build_voice_asr, resolve_voice_runtime_config


class FakeDetector:
    SAMPLE_RATE: ClassVar[int] = 16_000
    threshold = 0.5
    min_speech_ms = 250
    min_silence_ms = 500
    speech_pad_ms = 200

    def detect(self, audio: np.ndarray) -> VADResult:
        duration_ms = len(audio) / 16_000 * 1_000
        return VADResult(
            has_speech=True,
            speech_audio=audio,
            speech_duration_ms=duration_ms,
            original_duration_ms=duration_ms,
            speech_ratio=1.0,
            vad_latency_ms=0.0,
            n_speech_segments=1,
        )


class FakeQwenClient:
    DECODE_STRATEGY = "llm_decoder"

    def __init__(self, launch) -> None:
        self.model_key = launch.spec.key
        self.supports_avg_logprob = True
        self.calls: list[str] = []
        self.max_token_calls: list[int] = []
        self.close_calls = 0

    def transcribe(
        self,
        audio: np.ndarray,
        max_new_tokens: int = 128,
        *,
        context: str | None = None,
    ) -> ASRResult:
        self.max_token_calls.append(max_new_tokens)
        self.calls.append(context or "")
        return ASRResult(
            text="mở tài liệu attention",
            latency_ms=1.0,
            audio_duration_ms=len(audio) / 16_000 * 1_000,
            rtf=0.01,
            avg_logprob=0.0,
        )

    def runtime_metadata(self, max_new_tokens: int = 128) -> dict[str, object]:
        return {"max_new_tokens": max_new_tokens}

    def close(self) -> None:
        self.close_calls += 1


def _install_qwen_fakes(monkeypatch: pytest.MonkeyPatch) -> list[FakeQwenClient]:
    clients: list[FakeQwenClient] = []

    class FakeStore:
        def __init__(self, _root) -> None:
            pass

        def verify(self, _spec, *, deep: bool):
            assert deep is False
            return object()

    class FakeLaunch:
        @staticmethod
        def for_active(spec, _receipt):
            return SimpleNamespace(spec=spec)

    def client_factory(*, launch):
        client = FakeQwenClient(launch)
        clients.append(client)
        return client

    monkeypatch.setattr(voice_runtime, "QwenArtifactStore", FakeStore)
    monkeypatch.setattr(voice_runtime, "QwenServiceLaunch", FakeLaunch)
    monkeypatch.setattr(voice_runtime, "QwenASRServiceClient", client_factory)
    monkeypatch.setattr(
        voice_runtime,
        "load_strict_confidence_calibration",
        lambda identity: SimpleNamespace(
            identity=identity,
            min_avg_logprob=-0.5,
            max_compression_ratio=2.4,
            context_echo_min_contiguous_tokens=4,
        ),
    )
    return clients


def test_qwen_profiles_resolve_typed_release_and_reference(tmp_path) -> None:
    release = resolve_voice_runtime_config(profile_key="qwen-release", vault=tmp_path)
    reference = resolve_voice_runtime_config(profile_key="qwen-reference", vault=tmp_path)

    assert release.asr.engine is ASREngine.QWEN_SERVICE
    assert release.asr.artifact_role is ArtifactRole.RELEASE
    assert reference.asr.engine is ASREngine.QWEN_SERVICE
    assert reference.asr.artifact_role is ArtifactRole.REFERENCE


def test_default_runtime_vad_policy_matches_calibration_identity() -> None:
    assert compute_vad_policy_digest(FakeDetector()) == DEFAULT_VAD_POLICY_DIGEST


def test_qwen_final_uses_dynamic_context_and_partial_is_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    clients = _install_qwen_fakes(monkeypatch)
    config = resolve_voice_runtime_config(profile_key="qwen-release", vault=tmp_path)
    document = SimpleNamespace(
        path="wiki/learning/attention.md",
        title="Attention và Transformer",
        tags=("deep learning",),
        headings=(SimpleNamespace(text="Scaled dot product", line=12),),
    )
    catalog = SimpleNamespace(snapshot=lambda: SimpleNamespace(revision=7, documents=(document,)))
    runtime = _build_voice_asr(
        config,
        detector=FakeDetector(),
        knowledge_catalog=catalog,
        session_memory=None,
    )

    runtime.transcribe_partial(np.ones(160, dtype=np.float32))
    result = runtime.transcribe(np.ones(160, dtype=np.float32))

    assert len(clients) == 1
    assert clients[0].calls[0] == ""
    assert clients[0].max_token_calls[0] == 64
    assert "Attention và Transformer" in clients[0].calls[1]
    assert result.context_digest == runtime.last_context.digest
    assert any(item.startswith("vault:7:") for item in result.context_provenance)
    runtime.close()
    runtime.close()
    assert clients[0].close_calls == 1


def test_missing_qwen_calibration_blocks_before_service_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    started = False

    def reject_calibration(identity):
        raise ASRCalibrationNotReady(identity.digest)

    def forbidden_client(**_kwargs):
        nonlocal started
        started = True
        raise AssertionError("service must not start")

    monkeypatch.setattr(voice_runtime, "load_strict_confidence_calibration", reject_calibration)
    monkeypatch.setattr(voice_runtime, "QwenASRServiceClient", forbidden_client)
    config = resolve_voice_runtime_config(profile_key="qwen-release", vault=tmp_path)

    with pytest.raises(ASRCalibrationNotReady):
        _build_voice_asr(
            config,
            detector=FakeDetector(),
            knowledge_catalog=None,
            session_memory=None,
        )
    assert started is False


def test_calibration_lookup_requires_the_full_canonical_identity(tmp_path) -> None:
    identity = qwen_calibration_identity(QWEN_RELEASE_ARTIFACT)
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "calibrations": {
                    identity.digest: {
                        "identity": identity.payload,
                        "created_at_utc": "2026-08-02T00:00:00Z",
                        "recommended_thresholds": {
                            "min_avg_logprob": -0.4,
                            "max_compression_ratio": 2.4,
                            "context_echo_min_contiguous_tokens": 4,
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = load_strict_confidence_calibration(identity, path)
    assert loaded.identity == identity

    changed = replace(identity, max_new_tokens=identity.max_new_tokens + 1)
    with pytest.raises(ASRCalibrationNotReady):
        load_strict_confidence_calibration(changed, path)


@pytest.mark.parametrize(
    ("field", "value"),
    (("min_avg_logprob", float("nan")), ("max_compression_ratio", 0.0)),
)
def test_calibration_lookup_rejects_unsafe_thresholds(
    tmp_path,
    field: str,
    value: float,
) -> None:
    identity = qwen_calibration_identity(QWEN_RELEASE_ARTIFACT)
    thresholds = {
        "min_avg_logprob": -0.4,
        "max_compression_ratio": 2.4,
        "context_echo_min_contiguous_tokens": 4,
    }
    thresholds[field] = value
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "calibrations": {
                    identity.digest: {
                        "identity": identity.payload,
                        "created_at_utc": "2026-08-02T00:00:00Z",
                        "recommended_thresholds": thresholds,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ASRCalibrationError, match="malformed"):
        load_strict_confidence_calibration(identity, path)


@pytest.mark.parametrize("created_at_utc", (None, 123, "2026-08-02T07:00:00+07:00"))
def test_calibration_lookup_rejects_invalid_creation_timestamp(
    tmp_path,
    created_at_utc: object,
) -> None:
    identity = qwen_calibration_identity(QWEN_RELEASE_ARTIFACT)
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "calibrations": {
                    identity.digest: {
                        "identity": identity.payload,
                        "created_at_utc": created_at_utc,
                        "recommended_thresholds": {
                            "min_avg_logprob": -0.4,
                            "max_compression_ratio": 2.4,
                            "context_echo_min_contiguous_tokens": 4,
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ASRCalibrationError, match="malformed"):
        load_strict_confidence_calibration(identity, path)
