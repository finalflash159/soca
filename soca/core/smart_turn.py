from __future__ import annotations

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

    def p_still_speaking(self, audio_window: np.ndarray) -> float:
        feats = self._fe(
            _truncate_or_pad(audio_window), sampling_rate=_SR, return_tensors="np",
            padding="max_length", max_length=_N_SECONDS * _SR, truncation=True,
            do_normalize=True,
        ).input_features.squeeze(0).astype(np.float32)
        feats = np.expand_dims(feats, axis=0)  # (1, 80, 800)
        outputs = cast(list[np.ndarray], self._session.run(None, {"input_features": feats}))
        prob_complete = float(outputs[0][0][0].item())
        return float(np.clip(1.0 - prob_complete, 0.0, 1.0))

    def warmup(self) -> None:
        """Pay ORT graph/first-call cost off the hot path (call at controller startup)."""
        self.p_still_speaking(np.zeros(_SR, dtype=np.float32))
