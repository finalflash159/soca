from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from soca.app.style.palette import ALT, st
from soca.asr.registry import ASR_MODEL_REGISTRY
from soca.core.profiles import (
    DEFAULT_VOICE_RUNTIME_PROFILE_KEY,
    VOICE_RUNTIME_PROFILES,
    VoiceRuntimeProfile,
    validate_voice_runtime_profiles,
)
from soca.llm.registry import LLM_MODEL_REGISTRY
from soca.tts.registry import TTS_MODEL_REGISTRY, TTSModelConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VAULT = Path.home() / "KnowledgeVault"
VALTEC_SOURCE_ENV = "VALTEC_TTS_SOURCE_DIR"


@dataclass(frozen=True)
class ComponentReadiness:
    status: str
    detail: str

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class RuntimeProfileReadiness:
    key: str
    description: str
    asr_model: str
    asr_status: ComponentReadiness
    llm_model: str
    llm_status: ComponentReadiness
    tts_model: str
    tts_voice: str | None
    tts_status: ComponentReadiness
    profile_status: str
    notes: str


def collect_runtime_profile_readiness() -> list[RuntimeProfileReadiness]:
    validation_errors = _group_validation_errors(validate_voice_runtime_profiles())
    rows: list[RuntimeProfileReadiness] = []

    for key in sorted(VOICE_RUNTIME_PROFILES):
        profile = VOICE_RUNTIME_PROFILES[key]
        errors = validation_errors.get(key, ())
        if errors:
            invalid = ComponentReadiness("invalid", "; ".join(errors))
            rows.append(
                RuntimeProfileReadiness(
                    key=key,
                    description=profile.description,
                    asr_model=profile.asr_model,
                    asr_status=invalid,
                    llm_model=profile.llm_model,
                    llm_status=invalid,
                    tts_model=profile.tts_model,
                    tts_voice=profile.tts_voice,
                    tts_status=invalid,
                    profile_status="invalid",
                    notes=_join_notes(profile.description, invalid.detail),
                )
            )
            continue

        asr_status = _asr_readiness(profile.asr_model)
        llm_status = _llm_readiness(profile.llm_model)
        tts_status = _tts_readiness(profile)
        status = _combine_status(asr_status, llm_status, tts_status)
        notes = _join_notes(profile.description, tts_status.detail)

        rows.append(
            RuntimeProfileReadiness(
                key=key,
                description=profile.description,
                asr_model=profile.asr_model,
                asr_status=asr_status,
                llm_model=profile.llm_model,
                llm_status=llm_status,
                tts_model=profile.tts_model,
                tts_voice=profile.tts_voice,
                tts_status=tts_status,
                profile_status=status,
                notes=notes,
            )
        )

    return rows


def render_profiles(console: Console, *, show_paths: bool = False) -> None:
    rows = collect_runtime_profile_readiness()
    table = Table(title="SoCa Runtime Profiles")
    table.add_column("Profile", style=st(ALT) or "none", no_wrap=True)
    table.add_column("Stack", overflow="fold")
    table.add_column("Voice", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Notes", overflow="fold")

    for row in rows:
        table.add_row(
            row.key,
            "\n".join(
                [
                    f"ASR: {_format_component(row.asr_model, row.asr_status)}",
                    f"LLM: {_format_component(row.llm_model, row.llm_status)}",
                    f"TTS: {_format_component(row.tts_model, row.tts_status)}",
                ]
            ),
            row.tts_voice or "default",
            row.profile_status,
            row.notes,
        )

    console.print(table)
    console.print()
    console.print("[bold]Profile details[/bold]")
    for row in rows:
        console.print(
            f"{row.key}: ASR={row.asr_model} ({row.asr_status.status}); "
            f"LLM={row.llm_model} ({row.llm_status.status}); "
            f"TTS={row.tts_model} voice={row.tts_voice or 'default'} "
            f"({row.tts_status.status}); status={row.profile_status}; "
            f"{row.notes}",
            highlight=False,
        )

    if show_paths:
        console.print(_profile_paths_table(rows))
        console.print()
        console.print("[bold]Profile artifact path details[/bold]")
        for row in rows:
            asr = ASR_MODEL_REGISTRY.get(row.asr_model)
            llm = LLM_MODEL_REGISTRY.get(row.llm_model)
            tts = TTS_MODEL_REGISTRY.get(row.tts_model)
            console.print(
                f"{row.key}: "
                f"ASR={asr.local_dir if asr else 'unknown'}; "
                f"LLM={llm.local_path if llm else 'unknown'}; "
                f"TTS={tts.local_dir if tts else 'unknown'}",
                highlight=False,
            )


def render_status(console: Console, *, vault: Path = DEFAULT_VAULT) -> None:
    rows = collect_runtime_profile_readiness()
    validation_errors = validate_voice_runtime_profiles()
    ready_count = sum(1 for row in rows if row.profile_status == "ok")
    status_table = Table(title="SoCa Status")
    status_table.add_column("Item", style=st(ALT) or "none")
    status_table.add_column("Status", no_wrap=True)
    status_table.add_column("Detail")

    status_table.add_row("Package", "ok", f"soca {package_version()}")
    status_table.add_row("Primary command", "ok", "uv run soca voice --profile baseline")
    status_table.add_row("Default profile", "ok", DEFAULT_VOICE_RUNTIME_PROFILE_KEY)
    status_table.add_row(
        "Runtime profiles",
        "ok" if not validation_errors else "invalid",
        f"{ready_count}/{len(rows)} ready; {len(validation_errors)} validation issue(s)",
    )
    status_table.add_row(
        "Knowledge vault",
        "ok" if vault.expanduser().is_dir() else "missing",
        str(vault.expanduser()),
    )

    console.print(status_table)


def package_version() -> str:
    try:
        return importlib.metadata.version("soca")
    except importlib.metadata.PackageNotFoundError:
        return "editable"


def _profile_paths_table(rows: list[RuntimeProfileReadiness]) -> Table:
    table = Table(title="Profile Artifact Paths")
    table.add_column("Profile", style=st(ALT) or "none", no_wrap=True)
    table.add_column("ASR path", overflow="fold")
    table.add_column("LLM path", overflow="fold")
    table.add_column("TTS path", overflow="fold")

    for row in rows:
        asr = ASR_MODEL_REGISTRY.get(row.asr_model)
        llm = LLM_MODEL_REGISTRY.get(row.llm_model)
        tts = TTS_MODEL_REGISTRY.get(row.tts_model)
        table.add_row(
            row.key,
            str(asr.local_dir if asr else "unknown"),
            str(llm.local_path if llm else "unknown"),
            str(tts.local_dir if tts else "unknown"),
        )

    return table


def _group_validation_errors(errors: list[str]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for error in errors:
        key, _, message = error.partition(":")
        grouped.setdefault(key.strip() or "global", []).append(message.strip() or error)
    return {key: tuple(messages) for key, messages in grouped.items()}


def _asr_readiness(model_key: str) -> ComponentReadiness:
    config = ASR_MODEL_REGISTRY[model_key]
    missing = [
        path.name
        for path in (config.encoder_path, config.decoder_path)
        if not path.exists()
    ]
    if missing:
        return ComponentReadiness(
            "missing",
            f"missing ASR artifact(s): {', '.join(missing)}; {config.download_command}",
        )
    return ComponentReadiness("ok", "ONNX encoder/decoder found")


def _llm_readiness(model_key: str) -> ComponentReadiness:
    config = LLM_MODEL_REGISTRY[model_key]
    if not config.local_path.exists():
        return ComponentReadiness(
            "missing",
            f"missing GGUF: {config.local_path.name}; {config.download_command}",
        )
    return ComponentReadiness("ok", "GGUF found")


def _tts_readiness(profile: VoiceRuntimeProfile) -> ComponentReadiness:
    config = TTS_MODEL_REGISTRY[profile.tts_model]
    voice = profile.tts_voice or config.default_voice

    if config.runner == "valtec":
        return _valtec_readiness()
    if config.runner == "piper":
        return _piper_readiness(config)
    if config.runner == "omnivoice":
        return _omnivoice_readiness(config, voice)
    if config.runner == "vieneu":
        return _package_readiness("vieneu", "uv sync --extra tts-vieneu")
    if config.runner == "mms_transformers":
        return _packages_readiness(("torch", "transformers"), "uv sync --extra tts")
    if config.runner == "kani":
        return _package_readiness("kani_tts", "uv pip install kani-tts-2")
    if config.runner == "f5":
        return _f5_readiness(config)
    if config.runner == "viettts_server":
        return ComponentReadiness(
            "warning",
            "external server runner; start VietTTS server before use",
        )
    if config.runner == "external_command":
        return _external_command_readiness(config)

    return ComponentReadiness("warning", f"unknown runner readiness for {config.runner}")


def _valtec_readiness() -> ComponentReadiness:
    source = Path(os.environ.get(VALTEC_SOURCE_ENV, REPO_ROOT / "external" / "valtec-tts"))
    required = (source / "infer.py", source / "src")
    if all(path.exists() for path in required):
        return ComponentReadiness("ok", f"valtec source found: {source}")
    return ComponentReadiness(
        "missing",
        f"missing valtec source checkout; expected {source}",
    )


def _piper_readiness(config: TTSModelConfig) -> ComponentReadiness:
    package = _package_readiness("piper", "uv sync --extra tts-piper")
    if not package.ok:
        return package

    if not config.hf_model_file or not config.hf_config_file:
        return ComponentReadiness("missing", "missing Piper artifact metadata")

    model_path = config.local_dir / config.hf_model_file
    config_path = config.local_dir / config.hf_config_file
    if model_path.exists() and config_path.exists():
        return ComponentReadiness("ok", "Piper package and ONNX artifacts found")
    return ComponentReadiness("missing", f"missing Piper ONNX artifacts under {config.local_dir}")


def _omnivoice_readiness(config: TTSModelConfig, voice: str) -> ComponentReadiness:
    package = _package_readiness("omnivoice", "uv pip install omnivoice")
    if not package.ok:
        return package

    if voice in {config.default_voice, "auto"} or voice in (config.voice_designs or {}):
        return ComponentReadiness("ok", "OmniVoice package found; using generated voice design")

    voice_dir = config.local_dir / "voices" / voice
    if (voice_dir / "prompt.pt").exists() and (voice_dir / "meta.json").exists():
        return ComponentReadiness("ok", f"saved OmniVoice prompt found: {voice}")

    return ComponentReadiness("missing", f"missing saved OmniVoice voice prompt: {voice_dir}")


def _f5_readiness(config: TTSModelConfig) -> ComponentReadiness:
    package = _package_readiness("f5_tts", "uv sync --extra tts-f5")
    if not package.ok:
        return package

    audio = os.environ.get(config.reference_audio_env_var or "")
    text = os.environ.get(config.reference_text_env_var or "")
    if audio and text and Path(audio).expanduser().exists():
        return ComponentReadiness("ok", "F5 package and reference audio/text configured")
    return ComponentReadiness("missing", "missing F5 reference audio/text env vars")


def _external_command_readiness(config: TTSModelConfig) -> ComponentReadiness:
    command = os.environ.get(config.command_env_var or "") or ""
    if command.strip():
        return ComponentReadiness("ok", f"external command configured via {config.command_env_var}")
    return ComponentReadiness("missing", f"missing external command env var {config.command_env_var}")


def _package_readiness(module_name: str, install_hint: str) -> ComponentReadiness:
    if importlib.util.find_spec(module_name):
        return ComponentReadiness("ok", f"package {module_name} available")
    return ComponentReadiness("missing", f"missing package {module_name}; {install_hint}")


def _packages_readiness(module_names: tuple[str, ...], install_hint: str) -> ComponentReadiness:
    missing = [name for name in module_names if not importlib.util.find_spec(name)]
    if not missing:
        return ComponentReadiness("ok", f"packages available: {', '.join(module_names)}")
    return ComponentReadiness("missing", f"missing package(s): {', '.join(missing)}; {install_hint}")


def _combine_status(*components: ComponentReadiness) -> str:
    if any(component.status == "invalid" for component in components):
        return "invalid"
    if any(component.status == "missing" for component in components):
        return "missing"
    if any(component.status == "warning" for component in components):
        return "warning"
    return "ok"


def _format_component(name: str, readiness: ComponentReadiness) -> str:
    return f"{name} [{readiness.status}]"


def _join_notes(description: str, tts_detail: str) -> str:
    if not tts_detail:
        return description
    return f"{description} TTS: {tts_detail}"
