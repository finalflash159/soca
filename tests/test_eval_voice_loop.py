from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from eval import eval_voice_loop
from eval.eval_voice_loop import (
    VoiceLoopPrompt,
    VoiceLoopSample,
    collect_samples,
    generate_audio_fixtures,
    load_prompts,
    missing_sample_paths,
    parse_profiles,
    run_profile_eval,
    summarize,
    write_outputs,
)
from eval.result_io import make_eval_run_paths
from soca.core.streaming import StreamingEvent
from soca.tts import TTSResult


def write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(1600, dtype=np.float32), 16000)


def test_load_prompts_reads_jsonl_and_respects_limit(tmp_path: Path) -> None:
    prompt_path = tmp_path / "voice_loop.jsonl"
    prompt_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {"id": "hello", "category": "free_chat", "text": "xin chào"},
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "id": "time",
                        "category": "tool_time",
                        "text": "mấy giờ rồi",
                        "tags": ["tool"],
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    prompts = load_prompts(prompt_path, limit=1)

    assert prompts == [
        VoiceLoopPrompt(
            prompt_id="hello",
            category="free_chat",
            text="xin chào",
        )
    ]


def test_collect_samples_maps_prompts_to_wav_paths(tmp_path: Path) -> None:
    prompts = [
        VoiceLoopPrompt(
            prompt_id="nutrition",
            category="coach_nutrition",
            text="bữa sáng ăn gì",
            tags=("nutrition",),
        )
    ]
    audio_dir = tmp_path / "audio"

    samples = collect_samples(audio_dir=audio_dir, audio_files=[], prompts=prompts)

    assert samples == [
        VoiceLoopSample(
            sample_id="nutrition",
            audio_path=(audio_dir / "nutrition.wav").resolve(),
            expected_text="bữa sáng ăn gì",
            category="coach_nutrition",
            tags=("nutrition",),
        )
    ]


def test_missing_sample_paths_reports_missing_audio(tmp_path: Path) -> None:
    existing = tmp_path / "hello.wav"
    missing = tmp_path / "missing.wav"
    write_wav(existing)

    assert missing_sample_paths(
        [
            VoiceLoopSample("hello", existing),
            VoiceLoopSample("missing", missing),
        ]
    ) == [missing]


def test_parse_profiles_defaults_and_dedupes() -> None:
    assert parse_profiles([], all_profiles=False) == ["baseline"]
    assert parse_profiles(["baseline,baseline"], all_profiles=False) == ["baseline"]


def test_parse_profiles_can_select_all_profiles() -> None:
    profiles = parse_profiles([], all_profiles=True)

    assert profiles == ["baseline"]


@pytest.mark.parametrize("profile", ["quality", "edge"])
def test_parse_profiles_rejects_removed_profiles(profile: str) -> None:
    with pytest.raises(ValueError, match="Unknown profile"):
        parse_profiles([profile], all_profiles=False)


def test_parse_profiles_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="Unknown profile"):
        parse_profiles(["nope"], all_profiles=False)


def test_parser_rejects_runtime_tts_model_override() -> None:
    with pytest.raises(SystemExit):
        eval_voice_loop.build_parser().parse_args(
            ["--profile", "baseline", "--tts-model", "other_tts"]
        )


def test_parser_rejects_fixture_tts_model_override() -> None:
    with pytest.raises(SystemExit):
        eval_voice_loop.build_parser().parse_args(
            ["--generate-fixtures", "--fixture-tts-model", "other_tts"]
        )


def test_parser_uses_headless_playback_by_default() -> None:
    args = eval_voice_loop.build_parser().parse_args([])

    assert args.no_playback is False
    assert args.playback is False
    assert args.first_clause is None  # use the profile value unless overridden


def test_parser_playback_and_first_clause_toggles() -> None:
    parser = eval_voice_loop.build_parser()

    assert parser.parse_args(["--playback"]).playback is True
    assert parser.parse_args(["--first-clause"]).first_clause is True
    assert parser.parse_args(["--no-first-clause"]).first_clause is False


def test_summarize_handles_empty_and_values() -> None:
    assert summarize([])["median"] is None

    stats = summarize([1.0, 2.0, 3.0])

    assert stats["median"] == pytest.approx(2.0)
    assert stats["p95"] == pytest.approx(3.0)


def test_generate_audio_fixtures_uses_selected_tts(monkeypatch, tmp_path: Path) -> None:
    class FakeEngine:
        def synthesize(self, text: str, voice: str | None = None) -> TTSResult:
            return TTSResult(
                text=text,
                audio=np.ones(800, dtype=np.float32) * 0.1,
                sample_rate=16000,
                latency_ms=5.0,
                audio_duration_ms=50.0,
                rtf=0.1,
                voice=voice or "NF",
                engine="fake",
            )

    monkeypatch.setattr(eval_voice_loop, "create_tts_engine", lambda *args, **kwargs: FakeEngine())

    rows = generate_audio_fixtures(
        [VoiceLoopPrompt("hello", "free_chat", "xin chào")],
        audio_dir=tmp_path,
        voice="NF",
    )

    assert rows[0]["status"] == "generated"
    assert (tmp_path / "hello.wav").exists()


def test_run_profile_eval_with_fake_runtime(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "hello.wav"
    write_wav(audio_path)

    class FakePipeline:
        def turn_streaming(self, audio, audio_sink=None):
            yield StreamingEvent(type="asr", text="xin chào", metadata={"rejection_reason": ""})
            yield StreamingEvent(
                type="runtime",
                text="Chào bạn.",
                metadata={
                    "route": "free_chat",
                    "blocked": False,
                    "used_tool": False,
                    "used_llm": True,
                    "citations": [],
                },
            )
            yield StreamingEvent(
                type="tts",
                text="Chào bạn.",
                tts=TTSResult(
                    text="Chào bạn.",
                    audio=np.zeros(800, dtype=np.float32),
                    sample_rate=16000,
                    latency_ms=8.0,
                    audio_duration_ms=50.0,
                    rtf=0.16,
                    voice="NF",
                    engine="fake",
                ),
                metadata={"ttfa_ms": 30.0},
            )
            yield StreamingEvent(
                type="audio",
                text="Chào bạn.",
                metadata={"playback_latency_ms": 0.0},
            )
            yield StreamingEvent(
                type="done",
                text="Chào bạn.",
                latency_ms=60.0,
                metadata={
                    "rejected": False,
                    "runtime_route": "free_chat",
                    "stage_latencies_ms": {
                        "asr": 10.0,
                        "runtime": 20.0,
                        "tts_0": 8.0,
                    },
                },
            )

    fake_bundle = SimpleNamespace(
        pipeline=FakePipeline(),
        memory_status="disabled",
        knowledge_status="disabled",
        asr_guard_status="confidence=disabled",
    )
    monkeypatch.setattr(eval_voice_loop, "build_voice_runtime", lambda config: fake_bundle)

    args = SimpleNamespace(
        asr_model=None,
        llm_model=None,
        voice=None,
        max_tokens=None,
        temperature=None,
        top_p=None,
        vault=tmp_path,
        no_memory=True,
        memory_chars=2200,
        profile_chars=900,
        session_chars=1300,
        session_turns=6,
        turn_chars=500,
        no_playback=False,
        playback=False,
        first_clause=None,
    )

    result = run_profile_eval(
        "baseline",
        [VoiceLoopSample("hello", audio_path, expected_text="xin chào")],
        args=args,
    )

    assert result["status"] == "ok"
    assert result["playback_sink"] == "NullAudioPlayer"
    assert result["ttfa_ms"]["median"] == pytest.approx(30.0)
    assert result["tts_ready_ttfa_ms"]["median"] == pytest.approx(30.0)
    assert result["total_latency_ms"]["median"] == pytest.approx(60.0)
    assert result["route_counts"] == {"free_chat": 1}
    assert result["rows"][0]["transcript"] == "xin chào"


def test_run_profile_eval_playback_selects_sounddevice(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "hi.wav"
    write_wav(audio_path)
    seen: dict[str, str] = {}

    class RecordingPipeline:
        def turn_streaming(self, audio, audio_sink=None):
            seen["sink"] = type(audio_sink).__name__
            yield StreamingEvent(type="asr", text="xin chào", metadata={"rejection_reason": ""})
            yield StreamingEvent(
                type="done",
                text="Chào bạn.",
                latency_ms=10.0,
                metadata={
                    "rejected": False,
                    "runtime_route": "free_chat",
                    "stage_latencies_ms": {},
                },
            )

    fake_bundle = SimpleNamespace(
        pipeline=RecordingPipeline(),
        memory_status="disabled",
        knowledge_status="disabled",
        asr_guard_status="disabled",
    )
    monkeypatch.setattr(eval_voice_loop, "build_voice_runtime", lambda config: fake_bundle)
    # Keep the audio device untouched: only stop() would reach it, and no session is opened.
    monkeypatch.setattr("soca.core.audio_out.sd.stop", lambda: None)

    args = SimpleNamespace(
        asr_model=None,
        llm_model=None,
        voice=None,
        max_tokens=None,
        temperature=None,
        top_p=None,
        vault=tmp_path,
        no_memory=True,
        memory_chars=2200,
        profile_chars=900,
        session_chars=1300,
        session_turns=6,
        turn_chars=500,
        no_playback=False,
        playback=True,
        first_clause=False,
    )

    result = run_profile_eval("baseline", [VoiceLoopSample("hi", audio_path)], args=args)

    assert seen["sink"] == "SoundDevicePlayer"
    assert result["playback_sink"] == "SoundDevicePlayer"


def test_run_profile_eval_forwards_dense_backend(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, str] = {}

    def capture_config(config):
        seen["backend"] = config.knowledge_dense_backend
        raise eval_voice_loop.TTSRuntimeUnavailableError("test capture")

    monkeypatch.setattr(eval_voice_loop, "build_voice_runtime", capture_config)
    args = SimpleNamespace(
        asr_model=None,
        llm_model=None,
        voice=None,
        max_tokens=None,
        temperature=None,
        top_p=None,
        first_clause=None,
        vault=tmp_path,
        no_memory=True,
        memory_chars=2200,
        profile_chars=900,
        session_chars=1300,
        session_turns=6,
        turn_chars=500,
        knowledge_retrieval_mode=None,
        knowledge_dense_backend="model2vec",
    )

    result = run_profile_eval("baseline", [], args=args)

    assert result["status"] == "skipped_unavailable"
    assert seen["backend"] == "model2vec"


def test_write_outputs_creates_report_and_latest(tmp_path: Path) -> None:
    run_paths = make_eval_run_paths(tmp_path, "voice_loop", "20260601_000000")
    sample = VoiceLoopSample(
        sample_id="hello",
        audio_path=tmp_path / "hello.wav",
        expected_text="xin chào",
        category="free_chat",
    )
    result = {
        "profile": "baseline",
        "config": {
            "asr_model": "phowhisper_base",
            "llm_model": "arcee_vylinh_3b_q4_k_m",
            "tts_model": "valtec_multispeaker",
            "tts_voice": "NF",
        },
        "status": "ok",
        "load_ms": 1.0,
        "ttfa_ms": {"median": 30.0, "p95": 30.0},
        "total_latency_ms": {"median": 60.0, "p95": 60.0},
        "reject_rate": 0.0,
        "error_rate": 0.0,
        "peak_memory_mb": 123.0,
        "rows": [
            {
                "id": "hello",
                "transcript": "xin chào",
                "runtime_route": "free_chat",
                "stage_latencies_ms": {"asr": 10.0, "runtime": 20.0, "tts_0": 8.0},
                "ttfa_ms": 30.0,
                "total_latency_ms": 60.0,
                "tts_chunks": 1,
                "status": "ok",
            }
        ],
    }

    json_path, md_path = write_outputs(
        [result],
        run_paths,
        samples=[sample],
        fixture_generation=[],
    )

    assert json_path.exists()
    assert md_path.exists()
    assert run_paths.latest_json_path.exists()
    assert run_paths.latest_md_path.exists()
    report = md_path.read_text(encoding="utf-8")
    assert "SoCa E2E Voice Loop Benchmark" in report
    assert "Playback: `NullAudioPlayer`" in report
