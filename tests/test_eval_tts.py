from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eval import eval_tts
from eval.eval_tts import (
    TTSModelEvalTarget,
    TTSPrompt,
    build_eval_targets,
    clipping_ratio,
    load_prompts,
    parse_model_list,
    parse_voice_map,
    run_model_eval,
    select_model_keys,
    summarize,
    summarize_prompt_coverage,
    target_from_worker_payload,
    validate_prompt_coverage,
    worker_payload,
)
from soca.tts import DEFAULT_TTS_MODEL_KEY, TIER_A_TTS_MODEL_KEYS, TTSResult
from soca.tts.errors import TTSRuntimeUnavailableError


def test_load_prompts_reads_jsonl_and_respects_limit(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {"id": "a", "category": "short", "text": "Xin chào"},
                    ensure_ascii=False,
                ),
                json.dumps(
                    {"id": "b", "category": "coach", "text": "Tập nhẹ thôi"},
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    prompts = load_prompts(prompt_path, limit=1)

    assert len(prompts) == 1
    assert prompts[0].prompt_id == "a"
    assert prompts[0].text == "Xin chào"
    assert prompts[0].tags == ()


def test_load_prompts_reads_tags(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        json.dumps(
            {
                "id": "a",
                "category": "short",
                "tags": ["assistant", "short"],
                "text": "Xin chào",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    prompts = load_prompts(prompt_path)

    assert prompts[0].tags == ("assistant", "short")


def test_load_prompts_rejects_empty_file(tmp_path: Path) -> None:
    prompt_path = tmp_path / "empty.jsonl"
    prompt_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="No TTS prompts"):
        load_prompts(prompt_path)


def test_parse_model_list_accepts_comma_separated_values() -> None:
    assert parse_model_list(["valtec_multispeaker,mms_tts_vie", "piper_vi_vivos_x_low"]) == [
        "valtec_multispeaker",
        "mms_tts_vie",
        "piper_vi_vivos_x_low",
    ]


def test_parse_model_list_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unknown TTS model key"):
        parse_model_list(["not_real"])


def test_parse_voice_map_accepts_model_voice_pairs() -> None:
    assert parse_voice_map(["valtec_multispeaker=NF", "omnivoice=auto"]) == {
        "valtec_multispeaker": "NF",
        "omnivoice": "auto",
    }


def test_parse_voice_map_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unknown TTS model key"):
        parse_voice_map(["not_real=voice"])


def test_select_model_keys_defaults_to_registry_default() -> None:
    assert select_model_keys([], tier_a=False) == [DEFAULT_TTS_MODEL_KEY]


def test_select_model_keys_includes_tier_a_and_dedupes() -> None:
    assert select_model_keys(["valtec_multispeaker"], tier_a=True) == list(TIER_A_TTS_MODEL_KEYS)


def test_select_model_keys_can_select_all_registry_models() -> None:
    assert "omnivoice" in select_model_keys([], tier_a=False, all_models=True)


def test_build_eval_targets_keeps_profile_voice_separate_from_registry_default() -> None:
    targets = build_eval_targets([], ["quality"], voice=None, voice_map={})

    assert targets == [
        TTSModelEvalTarget(
            model_key="omnivoice",
            profile_key="quality",
            requested_voice="emgai_dangiu",
            voice_source="profile",
        )
    ]


def test_build_eval_targets_rejects_global_voice_for_multiple_models() -> None:
    with pytest.raises(ValueError, match="exactly one model"):
        build_eval_targets(
            ["valtec_multispeaker", "mms_tts_vie"],
            [],
            voice="NF",
            voice_map={},
        )


def test_build_eval_targets_supports_per_model_voice_map() -> None:
    targets = build_eval_targets(
        ["valtec_multispeaker", "omnivoice"],
        [],
        voice=None,
        voice_map={"valtec_multispeaker": "SF", "omnivoice": "auto"},
    )

    assert targets == [
        TTSModelEvalTarget(
            model_key="valtec_multispeaker",
            requested_voice="SF",
            voice_source="voice_map",
        ),
        TTSModelEvalTarget(
            model_key="omnivoice",
            requested_voice="auto",
            voice_source="voice_map",
        ),
    ]


def test_worker_payload_round_trips_target() -> None:
    target = TTSModelEvalTarget(
        model_key="omnivoice",
        profile_key="quality",
        requested_voice="emgai_dangiu",
        voice_source="profile",
    )

    assert target_from_worker_payload(worker_payload(target)) == target


def test_summarize_returns_distribution() -> None:
    stats = summarize([1.0, 2.0, 3.0])

    assert stats["mean"] == pytest.approx(2.0)
    assert stats["median"] == pytest.approx(2.0)
    assert stats["p95"] == pytest.approx(3.0)


def test_clipping_ratio_counts_samples_near_full_scale() -> None:
    audio = np.array([0.0, 0.5, 0.995, -1.0], dtype=np.float32)

    assert clipping_ratio(audio) == pytest.approx(0.5)


def test_prompt_coverage_reports_missing_required_categories() -> None:
    coverage = summarize_prompt_coverage(
        [TTSPrompt(prompt_id="a", category="short", text="Xin chào", tags=("assistant",))]
    )

    assert coverage["total_prompts"] == 1
    assert coverage["categories"] == {"short": 1}
    assert "nutrition" in coverage["missing_required_categories"]


def test_validate_prompt_coverage_accepts_default_corpus() -> None:
    prompts = load_prompts(eval_tts.DEFAULT_PROMPT_PATH)

    validate_prompt_coverage(prompts)


def test_run_model_eval_with_fake_engine(monkeypatch) -> None:
    class FakeEngine:
        def synthesize(self, text: str, voice: str | None = None) -> TTSResult:
            audio = np.ones(1600, dtype=np.float32) * 0.2
            return TTSResult(
                text=text,
                audio=audio,
                sample_rate=16000,
                latency_ms=20.0,
                audio_duration_ms=100.0,
                rtf=0.2,
                voice=voice or "default",
                engine="fake",
            )

        def list_voices(self) -> list[str]:
            return ["default"]

    monkeypatch.setattr(eval_tts, "create_tts_engine", lambda *args, **kwargs: FakeEngine())

    result = run_model_eval(
        "valtec_multispeaker",
        [TTSPrompt(prompt_id="a", category="short", text="Xin chào")],
        voice=None,
        skip_unavailable=True,
    )

    assert result is not None
    assert result["status"] == "ok"
    assert result["voices"] == [{"voice": "NF", "voice_source": "registry_default"}]
    assert result["non_empty_rate"] == pytest.approx(1.0)
    assert result["rtf"]["median"] == pytest.approx(0.2)


def test_run_model_eval_can_skip_unavailable(monkeypatch) -> None:
    def raise_unavailable(*args, **kwargs):
        raise TTSRuntimeUnavailableError("missing package")

    monkeypatch.setattr(eval_tts, "create_tts_engine", raise_unavailable)

    result = run_model_eval(
        "vieneu_v2_turbo",
        [TTSPrompt(prompt_id="a", category="short", text="Xin chào")],
        voice=None,
        skip_unavailable=True,
    )

    assert result is not None
    assert result["status"] == "skipped_unavailable"


def test_run_model_eval_can_skip_unavailable_from_list_voices(monkeypatch) -> None:
    class FakeEngine:
        def list_voices(self) -> list[str]:
            raise TTSRuntimeUnavailableError("server down")

        def synthesize(self, text: str, voice: str | None = None) -> TTSResult:
            raise AssertionError("synthesize should not run")

    monkeypatch.setattr(eval_tts, "create_tts_engine", lambda *args, **kwargs: FakeEngine())

    result = run_model_eval(
        "viet_tts_onnx",
        [TTSPrompt(prompt_id="a", category="short", text="Xin chào")],
        skip_unavailable=True,
    )

    assert result is not None
    assert result["status"] == "skipped_unavailable"
    assert "server down" in result["skip_reason"]
