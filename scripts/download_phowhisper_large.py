"""Download PhoWhisper-large ONNX into models/phowhisper-large-onnx.

huuquyet/PhoWhisper-large stores fp32 weights as external data (~6.2 GB) but also
ships self-contained int8 files. We map whichever variant is chosen onto the
standard ``encoder_model.onnx`` / ``decoder_model.onnx`` names that VietnameseASR
loads, so no runtime plumbing changes are needed.

    uv run python scripts/download_phowhisper_large.py                 # int8, ~1.6 GB
    uv run python scripts/download_phowhisper_large.py --precision fp32  # ~6.2 GB
"""

from __future__ import annotations

import shutil
from pathlib import Path

import click
from huggingface_hub import hf_hub_download
from rich.console import Console

from soca.asr.registry import get_asr_model_config

console = Console()

SUPPORT_FILES = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "normalizer.json",
    "added_tokens.json",
    "special_tokens_map.json",
]

# repo filename(s) -> local filename under onnx/. fp32 keeps external .onnx_data,
# which onnxruntime auto-loads from the sibling file.
ONNX_FILES = {
    "quantized": {
        "onnx/encoder_model_quantized.onnx": "encoder_model.onnx",
        "onnx/decoder_model_quantized.onnx": "decoder_model.onnx",
    },
    "fp32": {
        "onnx/encoder_model.onnx": "encoder_model.onnx",
        "onnx/encoder_model.onnx_data": "encoder_model.onnx_data",
        "onnx/decoder_model.onnx": "decoder_model.onnx",
        "onnx/decoder_model.onnx_data": "decoder_model.onnx_data",
    },
}


def _fetch(repo: str, repo_file: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(repo_id=repo, filename=repo_file)
    shutil.copyfile(cached, dest)
    console.print(f"  ✓ {repo_file} → {dest.relative_to(dest.parents[2])}")


@click.command()
@click.option(
    "--precision", default="quantized",
    type=click.Choice(["quantized", "fp32"]),
    help="quantized = int8 self-contained (~1.6 GB); fp32 = full weights (~6.2 GB).",
)
def main(precision: str) -> None:
    config = get_asr_model_config("phowhisper_large")
    model_dir = config.local_dir
    onnx_dir = model_dir / "onnx"
    repo = config.hf_repo

    console.print(f"[bold]Downloading {repo} ({precision})[/bold] → {model_dir}")

    console.print("Support files:")
    for name in SUPPORT_FILES:
        try:
            _fetch(repo, name, model_dir / name)
        except Exception as exc:  # optional tokenizer files vary per repo
            console.print(f"  [yellow]skip {name}: {type(exc).__name__}[/yellow]")

    console.print("ONNX weights:")
    for repo_file, local_name in ONNX_FILES[precision].items():
        _fetch(repo, repo_file, onnx_dir / local_name)

    console.print(
        f"\n[green]✓ PhoWhisper-large ({precision}) ready at {model_dir}[/green]\n"
        f"  Benchmark: uv run python -m local.eval_table7 --model phowhisper_large ..."
    )


if __name__ == "__main__":
    main()
