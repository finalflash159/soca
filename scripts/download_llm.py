from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

from huggingface_hub import hf_hub_download
from rich.console import Console
from rich.table import Table

from soca.llm.registry import (
    LLM_MODEL_REGISTRY,
    PROFILE_MODEL_KEYS,
    LLMModelConfig,
    get_model_config,
    get_profile_configs,
)

console = Console()


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.2f} GB"


def dedupe_configs(configs: Iterable[LLMModelConfig]) -> list[LLMModelConfig]:
    seen: set[str] = set()
    result: list[LLMModelConfig] = []
    for config in configs:
        if config.model_key in seen:
            continue
        seen.add(config.model_key)
        result.append(config)
    return result


def select_configs(model_keys: Sequence[str], profile: str | None) -> list[LLMModelConfig]:
    configs: list[LLMModelConfig] = []
    if profile is not None:
        configs.extend(get_profile_configs(profile))
    configs.extend(get_model_config(model_key) for model_key in model_keys)
    if not configs:
        configs.extend(get_profile_configs("minimal"))
    return dedupe_configs(configs)


def download_model(config: LLMModelConfig) -> Path:
    config.local_dir.mkdir(parents=True, exist_ok=True)
    console.print(
        f"[bold]Downloading[/bold] {config.model_key}\n"
        f"  repo: {config.hf_repo}\n"
        f"  file: {config.filename}\n"
        f"  dest: {config.local_dir}"
    )
    path = Path(
        hf_hub_download(
            repo_id=config.hf_repo,
            filename=config.filename,
            local_dir=str(config.local_dir),
        )
    )
    size = path.stat().st_size if path.exists() else 0
    console.print(f"[green]Done[/green] {path} ({format_bytes(size)})")
    return path


def print_model_table(configs: Sequence[LLMModelConfig]) -> None:
    table = Table(title="SoCa LLM Models")
    table.add_column("Key", style="cyan")
    table.add_column("Role")
    table.add_column("Repo")
    table.add_column("File")
    table.add_column("Exists", justify="center")

    for config in configs:
        table.add_row(
            config.model_key,
            config.role,
            config.hf_repo,
            config.filename,
            "yes" if config.local_path.exists() else "no",
        )

    console.print(table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download SoCa LLM GGUF artifacts.")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        choices=sorted(LLM_MODEL_REGISTRY),
        help="Model key to download. May be passed multiple times.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_MODEL_KEYS),
        help="Download a registry profile: minimal, bakeoff, or full.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List selected models without downloading.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configs = select_configs(args.model, args.profile)

    if args.list:
        print_model_table(configs)
        return 0

    print_model_table(configs)
    for config in configs:
        download_model(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
