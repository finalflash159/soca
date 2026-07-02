from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from soca.app import run_voice_loop
from soca.app.profiles import render_profiles, render_status
from soca.app.text_chat import run_text_chat
from soca.app.text_runtime import TextRuntimeConfig, resolve_text_runtime_config, run_text_ask
from soca.asr.registry import ASR_MODEL_REGISTRY, DEFAULT_ASR_MODEL_KEY
from soca.core import (
    DEFAULT_VOICE_RUNTIME_PROFILE_KEY,
    VOICE_RUNTIME_PROFILES,
    resolve_voice_runtime_config,
)
from soca.llm.registry import DEFAULT_LLM_MODEL_KEY, LLM_MODEL_REGISTRY
from soca.tts.registry import TTS_MODEL_REGISTRY

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
    """SoCa local Vietnamese voice assistant toolkit."""


@main.command()
@click.option(
    "--vault",
    type=click.Path(path_type=Path),
    default=Path.home() / "KnowledgeVault",
    show_default=True,
    help="Knowledge vault root to check.",
)
def status(vault: Path) -> None:
    """Show quick SoCa CLI/runtime readiness without loading models."""
    render_status(console, vault=vault)


@main.command("profiles")
@click.option(
    "--show-paths",
    is_flag=True,
    help="Also show registry artifact paths for each profile.",
)
def profiles_command(show_paths: bool) -> None:
    """Show configured voice runtime profiles without loading models."""
    render_profiles(console, show_paths=show_paths)


@main.command("ask")
@click.argument("text", nargs=-1, required=True)
@click.option(
    "--profile",
    default=DEFAULT_VOICE_RUNTIME_PROFILE_KEY,
    type=click.Choice(sorted(VOICE_RUNTIME_PROFILES)),
    show_default=True,
    help="Runtime profile whose LLM defaults are used.",
)
@click.option(
    "--llm-model",
    default=None,
    type=click.Choice(sorted(LLM_MODEL_REGISTRY)),
    help="Override the selected profile LLM for free-chat / knowledge-LLM routes.",
)
@click.option(
    "--vault",
    type=click.Path(path_type=Path),
    default=Path.home() / "KnowledgeVault",
    show_default=True,
    help="Knowledge vault root containing wiki/ and memory/profile.md.",
)
@click.option("--no-memory", is_flag=True, help="Disable profile/session memory.")
@click.option("--no-llm", is_flag=True, help="Run tool/guardrail-only without loading LLM.")
@click.option("--max-tokens", type=int, default=160, show_default=True)
@click.option("--temperature", type=float, default=0.2, show_default=True)
@click.option("--top-p", type=float, default=0.95, show_default=True)
@click.option("--knowledge-limit", type=int, default=3, show_default=True)
@click.option("--memory-chars", type=int, default=2200, show_default=True)
@click.option("--profile-chars", type=int, default=900, show_default=True)
@click.option("--session-chars", type=int, default=1300, show_default=True)
@click.option("--session-turns", type=int, default=6, show_default=True)
@click.option("--turn-chars", type=int, default=500, show_default=True)
@click.option("--trace/--no-trace", default=False, show_default=True)
@click.option("--usage", is_flag=True, help="Show LLM token/latency usage after the turn.")
def ask(
    text: tuple[str, ...],
    profile: str,
    llm_model: str | None,
    vault: Path,
    no_memory: bool,
    no_llm: bool,
    max_tokens: int,
    temperature: float,
    top_p: float,
    knowledge_limit: int,
    memory_chars: int,
    profile_chars: int,
    session_chars: int,
    session_turns: int,
    turn_chars: int,
    trace: bool,
    usage: bool,
) -> None:
    """Run one text-only SoCa turn without ASR/TTS."""
    config = build_text_runtime_config(
        profile=profile,
        llm_model=llm_model,
        vault=vault,
        no_memory=no_memory,
        no_llm=no_llm,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        knowledge_limit=knowledge_limit,
        memory_chars=memory_chars,
        profile_chars=profile_chars,
        session_chars=session_chars,
        session_turns=session_turns,
        turn_chars=turn_chars,
    )
    run_text_ask(" ".join(text), config, console=console, show_trace=trace, show_usage=usage)


@main.command("chat")
@click.option(
    "--profile",
    default=DEFAULT_VOICE_RUNTIME_PROFILE_KEY,
    type=click.Choice(sorted(VOICE_RUNTIME_PROFILES)),
    show_default=True,
    help="Runtime profile whose LLM defaults are used.",
)
@click.option(
    "--llm-model",
    default=None,
    type=click.Choice(sorted(LLM_MODEL_REGISTRY)),
    help="Override the selected profile LLM for free-chat / knowledge-LLM routes.",
)
@click.option(
    "--vault",
    type=click.Path(path_type=Path),
    default=Path.home() / "KnowledgeVault",
    show_default=True,
    help="Knowledge vault root containing wiki/ and memory/profile.md.",
)
@click.option("--no-memory", is_flag=True, help="Disable profile/session memory.")
@click.option("--no-llm", is_flag=True, help="Run tool/guardrail-only without loading LLM.")
@click.option("--max-tokens", type=int, default=160, show_default=True)
@click.option("--temperature", type=float, default=0.2, show_default=True)
@click.option("--top-p", type=float, default=0.95, show_default=True)
@click.option("--knowledge-limit", type=int, default=3, show_default=True)
@click.option("--memory-chars", type=int, default=2200, show_default=True)
@click.option("--profile-chars", type=int, default=900, show_default=True)
@click.option("--session-chars", type=int, default=1300, show_default=True)
@click.option("--session-turns", type=int, default=6, show_default=True)
@click.option("--turn-chars", type=int, default=500, show_default=True)
@click.option("--trace/--no-trace", default=False, show_default=True)
@click.option("--usage", is_flag=True, help="Show per-turn usage line; /usage shows session totals.")
@click.pass_context
def chat(
    ctx: click.Context,
    profile: str,
    llm_model: str | None,
    vault: Path,
    no_memory: bool,
    no_llm: bool,
    max_tokens: int,
    temperature: float,
    top_p: float,
    knowledge_limit: int,
    memory_chars: int,
    profile_chars: int,
    session_chars: int,
    session_turns: int,
    turn_chars: int,
    trace: bool,
    usage: bool,
) -> None:
    """Run an interactive text chat session without ASR/TTS."""
    config = build_text_runtime_config(
        profile=profile,
        llm_model=llm_model,
        vault=vault,
        no_memory=no_memory,
        no_llm=no_llm,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        knowledge_limit=knowledge_limit,
        memory_chars=memory_chars,
        profile_chars=profile_chars,
        session_chars=session_chars,
        session_turns=session_turns,
        turn_chars=turn_chars,
    )
    ctx.exit(run_text_chat(config, console=console, show_trace=trace, show_usage=usage))


@main.command(
    "ui",
    epilog=(
        "\b\nQuick examples:\n"
        "  uv run --extra ui soca ui status --no-model\n"
        "  uv run --extra ui soca ui chat\n"
        "  uv run --extra ui soca ui voice quality"
    ),
)
@click.argument(
    "quick_mode",
    required=False,
    type=click.Choice(["status", "chat", "voice"]),
)
@click.argument(
    "quick_profile",
    required=False,
    type=click.Choice(sorted(VOICE_RUNTIME_PROFILES)),
)
@click.option(
    "--mode",
    "mode_option",
    default=None,
    type=click.Choice(["status", "chat", "voice"]),
    hidden=True,
    help="Compatibility option. Prefer positional MODE: soca ui chat.",
)
@click.option(
    "--profile",
    "profile_option",
    default=None,
    type=click.Choice(sorted(VOICE_RUNTIME_PROFILES)),
    hidden=True,
    help="Compatibility option. Prefer positional PROFILE: soca ui voice quality.",
)
@click.option(
    "--llm-model",
    default=None,
    type=click.Choice(sorted(LLM_MODEL_REGISTRY)),
    hidden=True,
    help="Override the selected profile LLM for chat mode.",
)
@click.option(
    "--vault",
    type=click.Path(path_type=Path),
    default=Path.home() / "KnowledgeVault",
    show_default=True,
    help="Knowledge vault root containing wiki/ and memory/profile.md.",
)
@click.option("--no-memory", is_flag=True, help="Disable profile/session memory.")
@click.option("--no-llm", is_flag=True, help="Run chat mode tool/guardrail-only without loading LLM.")
@click.option(
    "--no-model",
    is_flag=True,
    help="Do not load model runtimes. Useful for status mode and UI smoke tests.",
)
@click.option("--max-tokens", type=int, default=160, hidden=True)
@click.option("--temperature", type=float, default=0.2, hidden=True)
@click.option("--top-p", type=float, default=0.95, hidden=True)
@click.option("--knowledge-limit", type=int, default=3, hidden=True)
@click.option("--memory-chars", type=int, default=2200, hidden=True)
@click.option("--profile-chars", type=int, default=900, hidden=True)
@click.option("--session-chars", type=int, default=1300, hidden=True)
@click.option("--session-turns", type=int, default=6, hidden=True)
@click.option("--turn-chars", type=int, default=500, hidden=True)
@click.pass_context
def ui(
    ctx: click.Context,
    quick_mode: str | None,
    quick_profile: str | None,
    mode_option: str | None,
    profile_option: str | None,
    llm_model: str | None,
    vault: Path,
    no_memory: bool,
    no_llm: bool,
    no_model: bool,
    max_tokens: int,
    temperature: float,
    top_p: float,
    knowledge_limit: int,
    memory_chars: int,
    profile_chars: int,
    session_chars: int,
    session_turns: int,
    turn_chars: int,
) -> None:
    """Open the optional Textual cockpit.

    Quick form: soca ui [status|chat|voice] [profile]
    """
    mode = quick_mode or mode_option or "status"
    profile = quick_profile or profile_option or DEFAULT_VOICE_RUNTIME_PROFILE_KEY

    try:
        from soca.app.tui import run_tui
    except ModuleNotFoundError as exc:
        missing_name = getattr(exc, "name", "")
        if missing_name and not missing_name.startswith("textual"):
            raise
        raise click.ClickException(
            "Textual UI dependencies are missing. Install/run with: "
            "uv sync --extra ui  hoặc  uv run --extra ui soca ui --mode status --no-model"
        ) from exc

    try:
        voice_config = resolve_voice_runtime_config(
            profile_key=profile,
            llm_model=llm_model,
            vault=vault,
            no_memory=no_memory,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            memory_chars=memory_chars,
            profile_chars=profile_chars,
            session_chars=session_chars,
            session_turns=session_turns,
            turn_chars=turn_chars,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    config = build_text_runtime_config(
        profile=profile,
        llm_model=voice_config.llm_model,
        vault=vault,
        no_memory=no_memory,
        no_llm=no_llm or no_model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        knowledge_limit=knowledge_limit,
        memory_chars=memory_chars,
        profile_chars=profile_chars,
        session_chars=session_chars,
        session_turns=session_turns,
        turn_chars=turn_chars,
    )
    # Path B barge-in (AEC + VAD) lives in the duplex sink player, which the TUI
    # builds lazily the first time voice mode is entered (see _ensure_voice_controller).
    ctx.exit(
        run_tui(
            mode=mode,
            profile=profile,
            text_runtime=config,
            voice_runtime=voice_config,
            no_model=no_model,
        )
    )


def build_text_runtime_config(
    *,
    profile: str = DEFAULT_VOICE_RUNTIME_PROFILE_KEY,
    llm_model: str | None,
    vault: Path,
    no_memory: bool,
    no_llm: bool,
    max_tokens: int,
    temperature: float,
    top_p: float,
    knowledge_limit: int,
    memory_chars: int,
    profile_chars: int,
    session_chars: int,
    session_turns: int,
    turn_chars: int,
) -> TextRuntimeConfig:
    try:
        return resolve_text_runtime_config(
            profile_key=profile,
            llm_model=llm_model,
            vault=vault,
            no_memory=no_memory,
            no_llm=no_llm,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            knowledge_limit=knowledge_limit,
            memory_chars=memory_chars,
            profile_chars=profile_chars,
            session_chars=session_chars,
            session_turns=session_turns,
            turn_chars=turn_chars,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


@main.command()
@click.argument(
    "quick_profile",
    required=False,
    type=click.Choice(sorted(VOICE_RUNTIME_PROFILES)),
)
@click.option(
    "--llm-model",
    default=None,
    type=click.Choice(sorted(LLM_MODEL_REGISTRY)),
    help="Override the selected profile LLM.",
)
@click.option(
    "--vault",
    type=click.Path(path_type=Path),
    default=Path.home() / "KnowledgeVault",
    show_default=True,
    help="Knowledge vault root containing wiki/ and memory/profile.md.",
)
@click.option("--no-memory", is_flag=True, help="Disable profile/session memory.")
@click.option(
    "--no-model",
    is_flag=True,
    help="Do not load model runtimes (protocol smoke tests).",
)
@click.pass_context
def engine(
    ctx: click.Context,
    quick_profile: str | None,
    llm_model: str | None,
    vault: Path,
    no_memory: bool,
    no_model: bool,
) -> None:
    """Run the headless NDJSON engine (stdio protocol for external UIs).

    The Ink TUI spawns this process: commands in on stdin, events out on
    stdout, audio stays in-process.
    """
    from soca.app.engine import run_engine

    profile = quick_profile or DEFAULT_VOICE_RUNTIME_PROFILE_KEY
    try:
        voice_config = resolve_voice_runtime_config(
            profile_key=profile,
            llm_model=llm_model,
            vault=vault,
            no_memory=no_memory,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    text_config = build_text_runtime_config(
        profile=profile,
        llm_model=voice_config.llm_model,
        vault=vault,
        no_memory=no_memory,
        no_llm=no_model,
        max_tokens=160,
        temperature=0.2,
        top_p=0.95,
        knowledge_limit=3,
        memory_chars=2200,
        profile_chars=900,
        session_chars=1300,
        session_turns=6,
        turn_chars=500,
    )
    ctx.exit(
        run_engine(
            voice_config=voice_config,
            text_config=text_config,
            profile=profile,
            no_model=no_model,
        )
    )


@main.command(
    epilog=(
        "\b\nQuick examples:\n"
        "  uv run soca voice baseline\n"
        "  uv run --extra tts-omnivoice soca voice quality\n"
        "  uv run --extra tts-piper soca voice edge"
    )
)
@click.argument(
    "quick_profile",
    required=False,
    type=click.Choice(sorted(VOICE_RUNTIME_PROFILES)),
)
@click.option(
    "--profile",
    "profile_option",
    default=None,
    type=click.Choice(sorted(VOICE_RUNTIME_PROFILES)),
    hidden=True,
    help="Voice runtime profile to use before explicit model overrides.",
)
@click.option(
    "--asr-model",
    default=None,
    type=click.Choice(sorted(ASR_MODEL_REGISTRY)),
    hidden=True,
    help="Override the ASR registry key from the selected profile.",
)
@click.option(
    "--llm-model",
    default=None,
    type=click.Choice(sorted(LLM_MODEL_REGISTRY)),
    hidden=True,
    help="Override the LLM registry key from the selected profile.",
)
@click.option(
    "--tts-model",
    default=None,
    type=click.Choice(sorted(TTS_MODEL_REGISTRY)),
    hidden=True,
    help="Override the TTS registry key from the selected profile.",
)
@click.option("--voice", default=None, hidden=True, help="Override the TTS voice/speaker id.")
@click.option("--endpoint-silence-ms", type=int, default=None, hidden=True)
@click.option("--max-record-ms", type=int, default=None, hidden=True)
@click.option(
    "--vault",
    type=click.Path(path_type=Path),
    default=Path.home() / "KnowledgeVault",
    show_default=True,
    help="Knowledge vault root containing wiki/ and memory/profile.md.",
)
@click.option("--no-memory", is_flag=True, help="Disable profile/session memory.")
@click.option("--memory-chars", type=int, default=2200, hidden=True)
@click.option("--profile-chars", type=int, default=900, hidden=True)
@click.option("--session-chars", type=int, default=1300, hidden=True)
@click.option("--session-turns", type=int, default=6, hidden=True)
@click.option("--turn-chars", type=int, default=500, hidden=True)
@click.option("--max-tokens", type=int, default=None, hidden=True)
@click.option("--temperature", type=float, default=None, hidden=True)
@click.option("--top-p", type=float, default=None, hidden=True)
@click.option(
    "--no-speak-repairs",
    is_flag=True,
    help="Do not speak the conversation-repair follow-up when ASR rejects a turn.",
)
@click.option(
    "--no-speak-rejections",
    is_flag=True,
    hidden=True,
    help="Deprecated alias of --no-speak-repairs.",
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
@click.option("--usage", is_flag=True, help="Show ASR/LLM/TTS latency + token usage after each turn.")
@click.option(
    "--barge-in/--no-barge-in",
    "barge_in",
    default=True,
    show_default=True,
    help="Interrupt playback when you start speaking. Use --no-barge-in on speakers "
    "without echo cancellation, otherwise SoCa hears itself and self-interrupts.",
)
@click.pass_context
def voice(
    ctx: click.Context,
    quick_profile: str | None,
    profile_option: str | None,
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
    no_speak_repairs: bool,
    no_speak_rejections: bool,
    press_enter_to_record: bool,
    no_warmup: bool,
    usage: bool,
    barge_in: bool,
) -> None:
    """Run the interactive SoCa microphone voice loop.

    Quick form: soca voice [profile]
    """
    profile = quick_profile or profile_option or DEFAULT_VOICE_RUNTIME_PROFILE_KEY

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

    player = None
    if barge_in:
        # Path B: duplex AEC sink does playback + echo cancel + barge-in on one clock.
        from soca.core.duplex_aec_sink import DuplexAecSink

        player = DuplexAecSink()
    ctx.exit(
        run_voice_loop(
            config,
            no_speak_repairs=no_speak_repairs,
            no_speak_rejections=no_speak_rejections,
            press_enter_to_record=press_enter_to_record,
            warmup=not no_warmup,
            show_usage=usage,
            player=player,
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
    table = Table(title="SoCa ASR Registry")
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
    table = Table(title="SoCa LLM Registry")
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
