from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np

_SR = 16000
_N_SECONDS = 8
_MODEL_FILE = "smart-turn-v3.2-cpu.onnx"


# according to pipecat smart-turn document: https://github.com/pipecat-ai/smart-turn
def _truncate_or_pad(audio: np.ndarray) -> np.ndarray:
    """Keep the last 8s (pad zeros at the front) - mirrors pipecat audio_utils."""
    cap = _N_SECONDS * _SR
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(x) > cap:
        return x[-cap:]
    if len(x) < cap:
        return np.pad(x, (cap - len(x), 0))
    return x


class SmartTurnDetector:
    def __init__(self, model_dir: Path, *, providers: list[str] | None = None) -> None:
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        onnx_path = Path(model_dir) / _MODEL_FILE
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"Smart Turn model missing: {onnx_path}. Run scripts/download_smart_turn.py"
            )
        opts = ort.SessionOptions()
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(onnx_path), sess_options=opts, providers=providers or ["CPUExecutionProvider"]
        )
        self._fe = WhisperFeatureExtractor(chunk_length=_N_SECONDS)

    def _input_features(self, audio_window: np.ndarray) -> np.ndarray:
        return self._fe(
            _truncate_or_pad(audio_window), sampling_rate=_SR, return_tensors="np",
            padding="max_length", max_length=_N_SECONDS * _SR, truncation=True,
            do_normalize=True,
        ).input_features.squeeze(0).astype(np.float32)

    def p_complete_batch(self, audio_windows: Sequence[np.ndarray]) -> np.ndarray:
        """Return P(turn complete) for a non-empty batch using the production model."""
        if not audio_windows:
            raise ValueError("Smart Turn batch must contain at least one audio window")
        features = np.stack([self._input_features(audio) for audio in audio_windows])
        outputs = cast(
            list[np.ndarray], self._session.run(None, {"input_features": features})
        )
        probabilities = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if len(probabilities) != len(audio_windows):
            raise RuntimeError(
                "Smart Turn output batch mismatch: "
                f"expected {len(audio_windows)}, got {len(probabilities)}"
            )
        return np.clip(probabilities, 0.0, 1.0)

    def p_still_speaking(self, audio_window: np.ndarray) -> float:
        prob_complete = float(self.p_complete_batch([audio_window])[0])
        return float(np.clip(1.0 - prob_complete, 0.0, 1.0))

    def warmup(self) -> None:
        """Pay ORT graph/first-call cost off the hot path (call at controller startup)."""
        self.p_still_speaking(np.zeros(_SR, dtype=np.float32))
