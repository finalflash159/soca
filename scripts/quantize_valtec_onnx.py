from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

SELECTIVE_GRAPHS = ("text_encoder", "duration_predictor", "flow")


def quantize_selective(fp32_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    for name in SELECTIVE_GRAPHS:
        source = fp32_dir / f"{name}.onnx"
        destination = output_dir / f"{name}.onnx"
        quantize_dynamic(
            model_input=str(source),
            model_output=str(destination),
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        onnx.checker.check_model(onnx.load(destination), full_check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    quantize_selective(args.fp32_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
