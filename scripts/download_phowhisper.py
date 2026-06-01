from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.table import Table

from soca.asr.registry import (
    ASR_MODEL_REGISTRY,
    DEFAULT_ASR_MODEL_KEY,
    PHOWHISPER_ALLOW_PATTERNS,
    ASRModelConfig,
    get_asr_model_config,
    get_asr_profile_model_keys,
)

console = Console()


def dedupe_configs(configs: Sequence[ASRModelConfig]) -> list[ASRModelConfig]:
    seen: set[str] = set()
    result: list[ASRModelConfig] = []
    for config in configs:
        if config.model_key in seen:
            continue
        seen.add(config.model_key)
        result.append(config)
    return result


def select_configs(model_keys: Sequence[str], profile: str | None) -> list[ASRModelConfig]:
    selected: list[ASRModelConfig] = []
    if profile:
        selected.extend(get_asr_model_config(model_key) for model_key in get_asr_profile_model_keys(profile))
    selected.extend(get_asr_model_config(model_key) for model_key in model_keys)
    if not selected:
        selected.append(get_asr_model_config(DEFAULT_ASR_MODEL_KEY))
    return dedupe_configs(selected)


def download_model(config: ASRModelConfig, models_dir: Path | None = None) -> Path:
    from huggingface_hub import snapshot_download

    local_dir = (models_dir / config.local_dir_name) if models_dir else config.local_dir
    local_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"[bold]Downloading[/bold] {config.model_key}\n"
        f"  repo: {config.hf_repo}\n"
        f"  dest: {local_dir}"
    )
    snapshot_download(
        repo_id=config.hf_repo,
        local_dir=str(local_dir),
        allow_patterns=PHOWHISPER_ALLOW_PATTERNS,
    )
    return local_dir


def print_model_table(configs: Sequence[ASRModelConfig], models_dir: Path | None = None) -> None:
    table = Table(title="Soca PhoWhisper ONNX Models")
    table.add_column("Key", style="cyan")
    table.add_column("Role")
    table.add_column("Params", justify="right")
    table.add_column("Repo")
    table.add_column("Exists", justify="center")

    for config in configs:
        local_dir = (models_dir / config.local_dir_name) if models_dir else config.local_dir
        table.add_row(
            config.model_key,
            config.role,
            f"{config.params_m}M",
            config.hf_repo,
            "yes" if local_dir.exists() else "no",
        )

    console.print(table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download PhoWhisper ONNX model files.")
    parser.add_argument(
        "--model",
        action="append",
        choices=sorted(ASR_MODEL_REGISTRY),
        default=[],
        help="ASR model key to download. May be passed multiple times.",
    )
    parser.add_argument(
        "--profile",
        choices=["minimal", "bakeoff", "full"],
        default=None,
        help="Download an ASR profile. minimal=tiny, bakeoff=tiny/base/small, full=all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Deprecated shortcut for --profile full.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List selected models without downloading.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Directory where model folders are stored. Overrides --storage.",
    )
    parser.add_argument(
        "--storage",
        choices=["local", "drive"],
        default="local",
        help="Default storage target when --models-dir is not provided.",
    )
    parser.add_argument(
        "--drive-root",
        type=Path,
        default=Path("/content/drive/MyDrive/soca"),
        help="Google Drive project root used when --storage=drive.",
    )
    return parser.parse_args()


def resolve_models_dir(args: argparse.Namespace) -> Path | None:
    if args.models_dir is not None:
        return args.models_dir
    if args.storage == "drive":
        return args.drive_root / "models"
    return None


def main() -> None:
    args = parse_args()
    profile = "full" if args.all else args.profile
    models_dir = resolve_models_dir(args)
    configs = select_configs(args.model, profile)

    print_model_table(configs, models_dir=models_dir)
    if args.list:
        return

    downloaded = [download_model(config, models_dir=models_dir) for config in configs]

    console.print("[green]Done.[/green]")
    for path in downloaded:
        console.print(f"- {path}")


if __name__ == "__main__":
    main()
