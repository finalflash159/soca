from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from soca.app import run_voice_loop
from soca.asr.registry import ASR_MODEL_REGISTRY, DEFAULT_ASR_MODEL_KEY, get_asr_model_config
from soca.core import (
    DEFAULT_VOICE_RUNTIME_PROFILE_KEY,
    VOICE_RUNTIME_PROFILES,
    resolve_voice_runtime_config,
)
from soca.llm.registry import DEFAULT_LLM_MODEL_KEY, LLM_MODEL_REGISTRY, get_model_config
from soca.tts import TTS_MODEL_REGISTRY

console = Console()
REPO_ROOT = Path(__file__).resolve().parents[1]


def run_module(module: str, *args: str) -> None:
    """Run a Python module using the current interpreter."""
    cmd = [sys.executable, "-m", module, *args]
    raise SystemExit(subprocess.call(cmd, cwd=REPO_ROOT))


def run_script(script: str, *args: str) -> None:
    """Run a script file using the current interpreter."""
    cmd = [sys.executable, script, *args]
    raise SystemExit(subprocess.call(cmd, cwd=REPO_ROOT))


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="soca")
def main() -> None:
    """Soca local Vietnamese voice assistant toolkit."""


@main.command()
def status() -> None:
    """Show local artifact status."""
    default_asr = get_asr_model_config(DEFAULT_ASR_MODEL_KEY)
    default_llm = get_model_config(DEFAULT_LLM_MODEL_KEY)
    paths = {
        f"Default ASR ONNX ({DEFAULT_ASR_MODEL_KEY})": default_asr.local_dir,
        f"Default LLM GGUF ({DEFAULT_LLM_MODEL_KEY})": default_llm.local_path,
        "Noise manifest": REPO_ROOT / "data" / "noise_for_boh" / "manifest.jsonl",
        "FLEURS manifest": REPO_ROOT / "data" / "fleurs_vi" / "manifest.jsonl",
        "Runtime BoH": REPO_ROOT / "data" / "asr" / "vi_boh_v1.json",
        "Threshold calibration": REPO_ROOT / "data" / "asr" / "threshold_calibration.json",
    }

    table = Table(title="Soca Local Status")
    table.add_column("Artifact", style="cyan")
    table.add_column("Exists", justify="center")
    table.add_column("Path")

    for name, path in paths.items():
        table.add_row(name, "yes" if path.exists() else "no", str(path))

    console.print(table)


@main.command()
@click.option(
    "--profile",
    default=DEFAULT_VOICE_RUNTIME_PROFILE_KEY,
    type=click.Choice(sorted(VOICE_RUNTIME_PROFILES)),
    show_default=True,
    help="Voice runtime profile to use before explicit model overrides.",
)
@click.option(
    "--asr-model",
    default=None,
    type=click.Choice(sorted(ASR_MODEL_REGISTRY)),
    help="Override the ASR registry key from the selected profile.",
)
@click.option(
    "--llm-model",
    default=None,
    type=click.Choice(sorted(LLM_MODEL_REGISTRY)),
    help="Override the LLM registry key from the selected profile.",
)
@click.option(
    "--tts-model",
    default=None,
    type=click.Choice(sorted(TTS_MODEL_REGISTRY)),
    help="Override the TTS registry key from the selected profile.",
)
@click.option("--voice", default=None, help="Override the TTS voice/speaker id.")
@click.option("--endpoint-silence-ms", type=int, default=None)
@click.option("--max-record-ms", type=int, default=None)
@click.option(
    "--vault",
    type=click.Path(path_type=Path),
    default=Path.home() / "KnowledgeVault",
    show_default=True,
    help="Knowledge vault root containing wiki/ and memory/profile.md.",
)
@click.option("--no-memory", is_flag=True, help="Disable profile/session memory.")
@click.option("--memory-chars", type=int, default=2200, show_default=True)
@click.option("--profile-chars", type=int, default=900, show_default=True)
@click.option("--session-chars", type=int, default=1300, show_default=True)
@click.option("--session-turns", type=int, default=6, show_default=True)
@click.option("--turn-chars", type=int, default=500, show_default=True)
@click.option("--max-tokens", type=int, default=None)
@click.option("--temperature", type=float, default=None)
@click.option("--top-p", type=float, default=None)
@click.option(
    "--no-speak-rejections",
    is_flag=True,
    help="Do not speak the fallback response when ASR rejects a turn.",
)
@click.option(
    "--press-enter-to-record",
    is_flag=True,
    help="Wait for ENTER before each recorded turn. Useful for debugging.",
)
@click.option(
    "--no-warmup",
    is_flag=True,
    help="Skip ASR/LLM/TTS first-call warmup before listening.",
)
@click.pass_context
def voice(
    ctx: click.Context,
    profile: str,
    asr_model: str | None,
    llm_model: str | None,
    tts_model: str | None,
    voice: str | None,
    endpoint_silence_ms: int | None,
    max_record_ms: int | None,
    vault: Path,
    no_memory: bool,
    memory_chars: int,
    profile_chars: int,
    session_chars: int,
    session_turns: int,
    turn_chars: int,
    max_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    no_speak_rejections: bool,
    press_enter_to_record: bool,
    no_warmup: bool,
) -> None:
    """Run the interactive Soca microphone voice loop."""
    try:
        config = resolve_voice_runtime_config(
            profile_key=profile,
            asr_model=asr_model,
            llm_model=llm_model,
            tts_model=tts_model,
            tts_voice=voice,
            endpoint_silence_ms=endpoint_silence_ms,
            max_record_ms=max_record_ms,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            vault=vault,
            no_memory=no_memory,
            memory_chars=memory_chars,
            profile_chars=profile_chars,
            session_chars=session_chars,
            session_turns=session_turns,
            turn_chars=turn_chars,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    ctx.exit(
        run_voice_loop(
            config,
            no_speak_rejections=no_speak_rejections,
            press_enter_to_record=press_enter_to_record,
            warmup=not no_warmup,
        )
    )


@main.command("asr-smoke")
@click.option(
    "--model",
    default=DEFAULT_ASR_MODEL_KEY,
    type=click.Choice(sorted(ASR_MODEL_REGISTRY)),
    show_default=True,
)
def asr_smoke(model: str) -> None:
    """Run the local ASR smoke test on recorded sample audio."""
    run_script(str(REPO_ROOT / "scripts" / "smoke_test_asr.py"), "--model", model)


@main.command("asr-models")
def asr_models() -> None:
    """List registered PhoWhisper ONNX candidates."""
    table = Table(title="Soca ASR Registry")
    table.add_column("Key", style="cyan")
    table.add_column("Role")
    table.add_column("Params", justify="right")
    table.add_column("Exists", justify="center")
    table.add_column("Path")

    for config in ASR_MODEL_REGISTRY.values():
        table.add_row(
            config.model_key,
            config.role,
            f"{config.params_m}M",
            "yes" if config.local_dir.exists() else "no",
            str(config.local_dir),
        )

    console.print(table)


@main.command("llm-smoke")
@click.option(
    "--model",
    default=DEFAULT_LLM_MODEL_KEY,
    type=click.Choice(sorted(LLM_MODEL_REGISTRY)),
    show_default=True,
)
def llm_smoke(model: str) -> None:
    """Run the local llama.cpp LLM smoke test."""
    run_script(str(REPO_ROOT / "scripts" / "smoke_test_llm.py"), "--model", model)


@main.command("llm-models")
def llm_models() -> None:
    """List registered local LLM candidates."""
    table = Table(title="Soca LLM Registry")
    table.add_column("Key", style="cyan")
    table.add_column("Role")
    table.add_column("Prompt")
    table.add_column("Exists", justify="center")
    table.add_column("Path")

    for config in LLM_MODEL_REGISTRY.values():
        table.add_row(
            config.model_key,
            config.role,
            config.prompt_style,
            "yes" if config.local_path.exists() else "no",
            str(config.local_path),
        )

    console.print(table)


@main.command("benchmark-asr")
@click.option("--n-speech", default=50, type=int, show_default=True)
@click.option("--n-noise", default=20, type=int, show_default=True)
@click.option(
    "--providers",
    default="auto",
    type=click.Choice(["auto", "cpu"]),
    show_default=True,
)
def benchmark_asr(n_speech: int, n_noise: int, providers: str) -> None:
    """Run the Table VII-style ASR robustness benchmark."""
    run_module(
        "local.eval_table7",
        "--n-speech",
        str(n_speech),
        "--n-noise",
        str(n_noise),
        "--providers",
        providers,
    )


@main.command("calibrate-asr")
@click.option("--n-speech", default=200, type=int, show_default=True)
@click.option("--n-noise", default=50, type=int, show_default=True)
@click.option(
    "--providers",
    default="auto",
    type=click.Choice(["auto", "cpu"]),
    show_default=True,
)
def calibrate_asr(n_speech: int, n_noise: int, providers: str) -> None:
    """Calibrate ASR confidence thresholds."""
    run_module(
        "local.calibrate_asr_confidence",
        "--n-speech",
        str(n_speech),
        "--n-noise",
        str(n_noise),
        "--providers",
        providers,
    )
