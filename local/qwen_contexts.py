"""Context strings for Qwen3-ASR context injection.

Kept separate from predict_qwen.py (which imports torch at module scope) so
tooling that only needs these strings — like the calibrator's whisper-onnx
path — doesn't require torch to be installed.
"""

from __future__ import annotations

CONTEXTS: dict[str, str] = {
    "none": "",
    "tech": (
        "Cuộc hội thoại về lập trình. Giữ nguyên cách viết các thuật ngữ tiếng Anh: "
        "GitHub, PyTorch, TensorFlow, TypeScript, PostgreSQL, Docker, Kubernetes, "
        "ONNX Runtime, Redis, Kafka, Nginx, FastAPI, Flask, FAISS, llama.cpp, "
        "API, JSON, GGUF, embedding, transformer, inference, latency, quantize."
    ),
}
