from __future__ import annotations

import pytest

from scripts.demo_voice_loop import build_parser, resolve_runtime_args


def parse_args(argv: list[str]):
    return resolve_runtime_args(build_parser().parse_args(argv))


def test_demo_baseline_resolves_former_quality_stack_with_valtec() -> None:
    args = parse_args(["--profile", "baseline"])

    assert args.asr_model == "phowhisper_small"
    assert args.llm_model == "arcee_vylinh_3b_q4_k_m"
    assert not hasattr(args, "tts_model")
    assert args.voice == "NF"


@pytest.mark.parametrize("profile", ["quality", "edge"])
def test_demo_parser_rejects_removed_profiles(profile: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--profile", profile])


def test_demo_parser_rejects_tts_model_override() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--profile", "baseline", "--tts-model", "other_tts"])


def test_explicit_voice_override_wins() -> None:
    args = parse_args(["--profile", "baseline", "--voice", "SF"])

    assert args.voice == "SF"


def test_asr_and_llm_overrides_still_win() -> None:
    args = parse_args(
        [
            "--profile",
            "baseline",
            "--asr-model",
            "phowhisper_base",
            "--llm-model",
            "phogpt_4b_q4_k_m",
        ]
    )

    assert args.asr_model == "phowhisper_base"
    assert args.llm_model == "phogpt_4b_q4_k_m"


def test_profile_runtime_options_resolve() -> None:
    args = parse_args(["--profile", "baseline"])

    assert args.endpoint_silence_ms == 700
    assert args.max_record_ms == 10000
    assert args.max_tokens == 160
    assert args.temperature == 0.2
    assert args.top_p == 0.95


def test_runtime_option_overrides_win() -> None:
    args = parse_args(
        [
            "--profile",
            "baseline",
            "--endpoint-silence-ms",
            "900",
            "--max-record-ms",
            "12000",
            "--max-tokens",
            "96",
            "--temperature",
            "0.1",
            "--top-p",
            "0.9",
        ]
    )

    assert args.endpoint_silence_ms == 900
    assert args.max_record_ms == 12000
    assert args.max_tokens == 96
    assert args.temperature == 0.1
    assert args.top_p == 0.9
