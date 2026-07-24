"""Export a trusted Valtec checkpoint into four split ONNX graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import torch
from src.models.synthesizer import SynthesizerTrn
from src.text.symbols import (
    language_id_map,
    language_tone_start_map,
    symbols,
)

GRAPH_NAMES = ("text_encoder", "duration_predictor", "flow", "decoder")
VOICE_SET = {"NF", "SF", "NM1", "SM", "NM2"}


class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, model: SynthesizerTrn) -> None:
        super().__init__()
        self.model = model

    def forward(self, phone_ids, phone_lengths, tone_ids, language_ids, bert, ja_bert, speaker_id):
        g = self.model.emb_g(speaker_id).unsqueeze(-1)
        g_for_encoder = None if self.model.use_vc else g
        x, m_p, logs_p, x_mask = self.model.enc_p(
            phone_ids,
            phone_lengths,
            tone_ids,
            language_ids,
            bert,
            ja_bert,
            g=g_for_encoder,
        )
        return x, m_p, logs_p, x_mask, g


class DurationPredictorWrapper(torch.nn.Module):
    def __init__(self, model: SynthesizerTrn) -> None:
        super().__init__()
        self.model = model

    def forward(self, x, x_mask, g):
        return self.model.dp(x, x_mask, g=g)


class FlowWrapper(torch.nn.Module):
    def __init__(self, model: SynthesizerTrn) -> None:
        super().__init__()
        self.model = model

    def forward(self, z_p, y_mask, g):
        return self.model.flow(z_p, y_mask, g=g, reverse=True)


class DecoderWrapper(torch.nn.Module):
    def __init__(self, model: SynthesizerTrn) -> None:
        super().__init__()
        self.model = model

    def forward(self, z, g):
        return self.model.dec(z, g=g)


def _load_model(checkpoint_path: Path, config: dict, device: str) -> SynthesizerTrn:
    model = SynthesizerTrn(
        len(symbols),
        config["data"]["filter_length"] // 2 + 1,
        config["train"]["segment_size"] // config["data"]["hop_length"],
        n_speakers=config["data"]["n_speakers"],
        **config["model"],
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = {key.removeprefix("module."): value for key, value in checkpoint["model"].items()}
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    return model


def _export(module, inputs, output: Path, *, input_names, output_names, dynamic_axes, opset: int) -> None:
    torch.onnx.export(
        module,
        inputs,
        output,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        dynamo=False,
        do_constant_folding=True,
    )
    model = onnx.load(output)
    onnx.checker.check_model(model, full_check=True)


def export_all(checkpoint: Path, source_config: Path, output_dir: Path, *, opset: int) -> None:
    config = json.loads(source_config.read_text(encoding="utf-8"))
    speaker_map = config["data"].get("spk2id")
    if not isinstance(speaker_map, dict) or set(speaker_map) != VOICE_SET:
        raise ValueError("Source config must contain exactly NF/SF/NM1/SM/NM2 in data.spk2id")
    if len(set(int(value) for value in speaker_map.values())) != 5:
        raise ValueError("Source config speaker ids must be unique")
    output_dir.mkdir(parents=True, exist_ok=False)
    model = _load_model(checkpoint, config, "cpu")

    text_len = 12
    phone_ids = torch.arange(text_len, dtype=torch.long).unsqueeze(0) % len(symbols)
    phone_lengths = torch.tensor([text_len], dtype=torch.long)
    tone_ids = torch.zeros((1, text_len), dtype=torch.long)
    language_ids = torch.full((1, text_len), int(language_id_map["VI"]), dtype=torch.long)
    bert = torch.zeros((1, 1024, text_len), dtype=torch.float32)
    ja_bert = torch.zeros((1, 768, text_len), dtype=torch.float32)
    speaker_id = torch.tensor([int(speaker_map["NF"])], dtype=torch.long)

    encoder = TextEncoderWrapper(model)
    encoder_inputs = (phone_ids, phone_lengths, tone_ids, language_ids, bert, ja_bert, speaker_id)
    _export(
        encoder,
        encoder_inputs,
        output_dir / "text_encoder.onnx",
        input_names=["phone_ids", "phone_lengths", "tone_ids", "language_ids", "bert", "ja_bert", "speaker_id"],
        output_names=["x", "m_p", "logs_p", "x_mask", "g"],
        dynamic_axes={
            "phone_ids": {1: "text_len"}, "tone_ids": {1: "text_len"},
            "language_ids": {1: "text_len"}, "bert": {2: "text_len"},
            "ja_bert": {2: "text_len"}, "x": {2: "text_len"},
            "m_p": {2: "text_len"}, "logs_p": {2: "text_len"},
            "x_mask": {2: "text_len"},
        },
        opset=opset,
    )
    with torch.no_grad():
        x, m_p, logs_p, x_mask, g = encoder(*encoder_inputs)

    _export(
        DurationPredictorWrapper(model),
        (x, x_mask, g),
        output_dir / "duration_predictor.onnx",
        input_names=["x", "x_mask", "g"],
        output_names=["logw"],
        dynamic_axes={"x": {2: "text_len"}, "x_mask": {2: "text_len"}, "logw": {2: "text_len"}},
        opset=opset,
    )
    frame_len = max(text_len, 32)
    z_p = torch.zeros((1, m_p.shape[1], frame_len), dtype=torch.float32)
    y_mask = torch.ones((1, 1, frame_len), dtype=torch.float32)
    _export(
        FlowWrapper(model),
        (z_p, y_mask, g),
        output_dir / "flow.onnx",
        input_names=["z_p", "y_mask", "g"],
        output_names=["z"],
        dynamic_axes={"z_p": {2: "frame_len"}, "y_mask": {2: "frame_len"}, "z": {2: "frame_len"}},
        opset=opset,
    )
    _export(
        DecoderWrapper(model),
        (z_p, g),
        output_dir / "decoder.onnx",
        input_names=["z", "g"],
        output_names=["audio"],
        dynamic_axes={"z": {2: "frame_len"}, "audio": {2: "audio_len"}},
        opset=opset,
    )
    runtime_config = {
        "sample_rate": int(config["data"]["sampling_rate"]),
        "hop_length": int(config["data"]["hop_length"]),
        "add_blank": bool(config["data"].get("add_blank", True)),
        "symbol_to_id": {symbol: index for index, symbol in enumerate(symbols)},
        "language_id_map": {name: int(value) for name, value in language_id_map.items()},
        "tone_offset_vi": int(language_tone_start_map["VI"]),
        "speaker_id_map": {name: int(value) for name, value in speaker_map.items()},
    }
    (output_dir.parent / "tts_config.json").write_text(
        json.dumps(runtime_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--trust-checkpoint", action="store_true")
    args = parser.parse_args()
    if not args.trust_checkpoint:
        parser.error("--trust-checkpoint is required because export uses torch.load")
    export_all(args.checkpoint, args.config, args.output_dir, opset=args.opset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
