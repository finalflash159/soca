from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from soca.tts.valtec.frontend import ValtecModelInputs

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "external" / "valtec-tts"
WORKER = REPO_ROOT / "scripts" / "valtec_torch_worker.py"


@dataclass(frozen=True)
class TorchReferenceResult:
    audio: np.ndarray
    sample_rate: int


def synthesize_torch_reference(
    model_inputs: ValtecModelInputs,
    *,
    speaker_id: int,
    checkpoint: Path,
    config: Path,
    trust_checkpoint: bool,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    device: str = "cpu",
) -> TorchReferenceResult:
    if not trust_checkpoint:
        raise ValueError("Torch reference requires explicit trust_checkpoint=True")
    for path in (checkpoint, config, source_root / "src", WORKER):
        if not path.exists():
            raise FileNotFoundError(f"Missing Valtec Torch reference input: {path}")
    environment = os.environ.copy()
    old_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(source_root) if not old_pythonpath else os.pathsep.join((str(source_root), old_pythonpath))
    )
    with tempfile.TemporaryDirectory(prefix="soca_valtec_parity_") as temporary:
        root = Path(temporary)
        input_path = root / "inputs.npz"
        output_path = root / "torch.wav"
        np.savez_compressed(
            input_path,
            phone_ids=np.asarray([model_inputs.phone_ids], dtype=np.int64),
            tone_ids=np.asarray([model_inputs.tone_ids], dtype=np.int64),
            language_ids=np.asarray([model_inputs.language_ids], dtype=np.int64),
            speaker_id=np.asarray([speaker_id], dtype=np.int64),
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(WORKER),
                "--checkpoint", str(checkpoint),
                "--config", str(config),
                "--inputs", str(input_path),
                "--output", str(output_path),
                "--device", device,
            ],
            cwd=source_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Valtec Torch reference worker failed:\n"
                + (completed.stderr or completed.stdout)
            )
        audio, sample_rate = sf.read(output_path, dtype="float32", always_2d=False)
    return TorchReferenceResult(np.ascontiguousarray(audio), int(sample_rate))