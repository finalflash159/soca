from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eval import eval_tts
from eval.eval_tts import (
    TTSPrompt,
    ValtecEvalTarget,
    build_parser,
    build_valtec_eval_target,
    clipping_ratio,
    load_prompts,
    run_valtec_eval,
    summarize,
    summarize_prompt_coverage,
    target_from_worker_payload,
    validate_prompt_coverage,
    worker_payload,
)
from soca.tts import TTSResult
from soca.tts.errors import TTSRuntimeUnavailableError


@pytest.mark.parametrize(
    "args",
    [
        ["--model", "other_tts"],
        ["--tier-a"],
        ["--tier-b"],
        ["--all"],
        ["--voice-map", "valtec_multispeaker=SF"],
        ["--profile", "qwen-release"],
    ],
)
def test_parser_rejects_multi_model_selectors(args: list[str]) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(args)


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


def test_build_valtec_eval_target_uses_qwen_release_voice() -> None:
    assert build_valtec_eval_target(voice=None) == ValtecEvalTarget(
        profile_key="qwen-release",
        requested_voice="NF",
        voice_source="profile",
    )


def test_build_valtec_eval_target_accepts_explicit_voice() -> None:
    assert build_valtec_eval_target(voice="SF") == ValtecEvalTarget(
        profile_key="qwen-release",
        requested_voice="SF",
        voice_source="cli",
    )


def test_worker_payload_round_trips_target() -> None:
    target = ValtecEvalTarget(
        profile_key="qwen-release",
        requested_voice="SF",
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


def test_run_valtec_eval_with_fake_engine(monkeypatch) -> None:
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

    result = run_valtec_eval(
        None,
        [TTSPrompt(prompt_id="a", category="short", text="Xin chào")],
        voice=None,
        skip_unavailable=True,
    )

    assert result is not None
    assert result["status"] == "ok"
    assert result["model"] == "valtec_multispeaker"
    assert result["voices"] == [{"voice": "NF", "voice_source": "profile"}]
    assert result["non_empty_rate"] == pytest.approx(1.0)
    assert result["rtf"]["median"] == pytest.approx(0.2)


def test_run_valtec_eval_can_skip_unavailable(monkeypatch) -> None:
    def raise_unavailable(*args, **kwargs):
        raise TTSRuntimeUnavailableError("missing package")

    monkeypatch.setattr(eval_tts, "create_tts_engine", raise_unavailable)

    result = run_valtec_eval(
        None,
        [TTSPrompt(prompt_id="a", category="short", text="Xin chào")],
        voice=None,
        skip_unavailable=True,
    )

    assert result is not None
    assert result["status"] == "skipped_unavailable"


def test_run_valtec_eval_can_skip_unavailable_from_list_voices(monkeypatch) -> None:
    class FakeEngine:
        def list_voices(self) -> list[str]:
            raise TTSRuntimeUnavailableError("server down")

        def synthesize(self, text: str, voice: str | None = None) -> TTSResult:
            raise AssertionError("synthesize should not run")

    monkeypatch.setattr(eval_tts, "create_tts_engine", lambda *args, **kwargs: FakeEngine())

    result = run_valtec_eval(
        None,
        [TTSPrompt(prompt_id="a", category="short", text="Xin chào")],
        skip_unavailable=True,
    )

    assert result is not None
    assert result["status"] == "skipped_unavailable"
    assert "server down" in result["skip_reason"]
