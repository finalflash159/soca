from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from soca.app.style.palette import ALT, st
from soca.asr.calibration import (
    ASRCalibrationError,
    load_strict_confidence_calibration,
    qwen_calibration_identity,
)
from soca.asr.confidence_calibration import load_confidence_guard_calibration
from soca.asr.qwen_artifacts import QWEN_ARTIFACT_REGISTRY
from soca.asr.qwen_readiness import QwenReadinessState, inspect_qwen_readiness
from soca.asr.registry import ASR_MODEL_REGISTRY
from soca.asr.selection import ASREngine, ASRSelection
from soca.core.profiles import (
    DEFAULT_VOICE_RUNTIME_PROFILE_KEY,
    VOICE_RUNTIME_PROFILES,
    validate_voice_runtime_profiles,
)
from soca.knowledge.vault import default_vault_root
from soca.llm.registry import LLM_MODEL_REGISTRY
from soca.tts import VALTEC_TTS_CONFIG
from soca.tts.valtec.artifacts import (
    resolve_current_valtec_release,
    resolve_valtec_onnx_artifacts,
)

DEFAULT_VAULT = default_vault_root()


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
    tts_engine: str
    tts_voice: str | None
    tts_status: ComponentReadiness
    profile_status: str
    notes: str


def collect_runtime_profile_readiness() -> list[RuntimeProfileReadiness]:
    validation_errors = _group_validation_errors(validate_voice_runtime_profiles())
    rows: list[RuntimeProfileReadiness] = []

    for key in sorted(VOICE_RUNTIME_PROFILES):
        profile = VOICE_RUNTIME_PROFILES[key]
        # Whole-config invariant failures ("global") apply to every profile, so
        # surface them on each row instead of letting the table read all-ok.
        errors = validation_errors.get(key, ()) + validation_errors.get("global", ())
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
                    tts_engine=VALTEC_TTS_CONFIG.key,
                    tts_voice=profile.tts_voice,
                    tts_status=invalid,
                    profile_status="invalid",
                    notes=_join_notes(profile.description, invalid.detail),
                )
            )
            continue

        asr_status = asr_readiness(profile.asr)
        llm_status = _llm_readiness(profile.llm_model)
        tts_status = _valtec_readiness()
        status = _combine_status(asr_status, llm_status, tts_status)
        notes = _join_notes(
            profile.description,
            tts_status.detail,
            asr_detail=asr_status.detail if not asr_status.ok else "",
        )

        rows.append(
            RuntimeProfileReadiness(
                key=key,
                description=profile.description,
                asr_model=profile.asr_model,
                asr_status=asr_status,
                llm_model=profile.llm_model,
                llm_status=llm_status,
                tts_engine=VALTEC_TTS_CONFIG.key,
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
                    f"TTS: {_format_component(row.tts_engine, row.tts_status)}",
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
            f"TTS={row.tts_engine} voice={row.tts_voice or 'default'} "
            f"({row.tts_status.status}); status={row.profile_status}; "
            f"{row.notes}",
            highlight=False,
        )

    if show_paths:
        console.print(_profile_paths_table(rows))
        console.print()
        console.print("[bold]Profile artifact path details[/bold]")
        for row in rows:
            llm = LLM_MODEL_REGISTRY.get(row.llm_model)
            console.print(
                f"{row.key}: "
                f"ASR={_asr_path(VOICE_RUNTIME_PROFILES[row.key].asr)}; "
                f"LLM={llm.local_path if llm else 'unknown'}; "
                f"TTS={VALTEC_TTS_CONFIG.local_dir}",
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
    status_table.add_row("Primary command", "ok", "uv run soca voice")
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
        llm = LLM_MODEL_REGISTRY.get(row.llm_model)
        table.add_row(
            row.key,
            str(_asr_path(VOICE_RUNTIME_PROFILES[row.key].asr)),
            str(llm.local_path if llm else "unknown"),
            str(VALTEC_TTS_CONFIG.local_dir),
        )

    return table


def _group_validation_errors(errors: list[str]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for error in errors:
        # Profile-scoped errors are "<profile>: message"; anything without a
        # colon is a whole-config invariant (e.g. the profile-set check) and
        # must bucket under "global" so it is not lost against a profile name.
        if ":" in error:
            key, _, message = error.partition(":")
            key, message = key.strip() or "global", message.strip() or error
        else:
            key, message = "global", error
        grouped.setdefault(key, []).append(message)
    return {key: tuple(messages) for key, messages in grouped.items()}


def asr_readiness(selection: ASRSelection) -> ComponentReadiness:
    if selection.engine is ASREngine.QWEN_SERVICE:
        spec = QWEN_ARTIFACT_REGISTRY[selection.model_key]
        readiness = inspect_qwen_readiness(spec)
        if readiness.state is not QwenReadinessState.PROVISIONED:
            return ComponentReadiness(readiness.state.value, readiness.detail)
        try:
            calibration = load_strict_confidence_calibration(
                qwen_calibration_identity(spec)
            )
        except ASRCalibrationError as exc:
            return ComponentReadiness("not_ready", str(exc))
        return ComponentReadiness(
            "ok",
            f"artifact and calibration {calibration.identity.digest[:12]} verified",
        )

    config = ASR_MODEL_REGISTRY[selection.model_key]
    missing = [
        path.name for path in (config.encoder_path, config.decoder_path) if not path.exists()
    ]
    if missing:
        return ComponentReadiness(
            "missing",
            f"missing ASR artifact(s): {', '.join(missing)}; {config.download_command}",
        )
    if load_confidence_guard_calibration(selection.model_key) is None:
        return ComponentReadiness(
            "not_ready",
            f"confidence calibration is missing for {selection.model_key}",
        )
    return ComponentReadiness("ok", "ONNX encoder/decoder found")


def _asr_path(selection: ASRSelection) -> Path | str:
    if selection.engine is ASREngine.QWEN_SERVICE:
        return QWEN_ARTIFACT_REGISTRY[selection.model_key].model_path()
    return ASR_MODEL_REGISTRY[selection.model_key].local_dir


def _llm_readiness(model_key: str) -> ComponentReadiness:
    config = LLM_MODEL_REGISTRY[model_key]
    if not config.local_path.exists():
        return ComponentReadiness(
            "missing",
            f"missing GGUF: {config.local_path.name}; {config.download_command}",
        )
    return ComponentReadiness("ok", "GGUF found")


def _valtec_readiness() -> ComponentReadiness:
    try:
        release_root = resolve_current_valtec_release()
        artifacts = resolve_valtec_onnx_artifacts(release_root)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        return ComponentReadiness("missing", f"Valtec artifact is not ready: {exc}")
    return ComponentReadiness(
        "ok",
        f"Valtec {release_root.name}/{artifacts.precision} manifest verified",
    )


def _combine_status(*components: ComponentReadiness) -> str:
    for status in ("invalid", "missing", "not_ready", "unsupported", "warning"):
        if any(component.status == status for component in components):
            return status
    return "ok"


def _format_component(name: str, readiness: ComponentReadiness) -> str:
    return f"{name} [{readiness.status}]"


def _join_notes(description: str, tts_detail: str, *, asr_detail: str = "") -> str:
    details = [description]
    if asr_detail:
        details.append(f"ASR: {asr_detail}")
    if tts_detail:
        details.append(f"TTS: {tts_detail}")
    return " ".join(details)
