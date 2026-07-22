"""Trusted build/eval worker for deterministic Valtec Torch inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from src.models.synthesizer import SynthesizerTrn
from src.text.symbols import symbols


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model = SynthesizerTrn(
        len(symbols),
        config["data"]["filter_length"] // 2 + 1,
        config["train"]["segment_size"] // config["data"]["hop_length"],
        n_speakers=config["data"]["n_speakers"],
        **config["model"],
    ).to(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=args.device,
        weights_only=False,
    )
    state = {
        key.removeprefix("module."): value
        for key, value in checkpoint["model"].items()
    }
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Valtec checkpoint/model mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.eval()

    arrays = np.load(args.inputs, allow_pickle=False)
    phone_ids = torch.from_numpy(arrays["phone_ids"]).long().to(args.device)
    tone_ids = torch.from_numpy(arrays["tone_ids"]).long().to(args.device)
    language_ids = torch.from_numpy(arrays["language_ids"]).long().to(args.device)
    speaker_id = torch.from_numpy(arrays["speaker_id"]).long().to(args.device)
    lengths = torch.tensor([phone_ids.shape[1]], dtype=torch.long, device=args.device)
    bert = torch.zeros((1, 1024, phone_ids.shape[1]), device=args.device)
    ja_bert = torch.zeros((1, 768, phone_ids.shape[1]), device=args.device)
    with torch.inference_mode():
        audio, *_ = model.infer(
            phone_ids,
            lengths,
            speaker_id,
            tone_ids,
            language_ids,
            bert,
            ja_bert,
            sdp_ratio=0.0,
            noise_scale=0.0,
            noise_scale_w=0.0,
            length_scale=1.0,
        )
    waveform = audio[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, waveform, int(config["data"]["sampling_rate"]), subtype="FLOAT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())