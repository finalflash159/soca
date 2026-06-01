"""Shared paths and constants for the local D2.5 workflow.

This module is the single source of truth for local artifact locations.
Notebooks 01/02 hold their own copies for Colab; intentional duplication
because notebook config has Drive-specific branches we don't need here.
"""

from __future__ import annotations

from pathlib import Path

from soca.asr.registry import ASR_MODEL_REGISTRY, PHOWHISPER_ALLOW_PATTERNS

# Repo root resolved at import time. `local/` is one level below the root.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Heavy artifacts — all under data/ which is gitignored.
DATA_DIR = REPO_ROOT / "data"
NOISE_ROOT = DATA_DIR / "noise_for_boh"
NOISE_WAV_DIR = NOISE_ROOT / "wav"
NOISE_MANIFEST = NOISE_ROOT / "manifest.jsonl"
NOISE_CONFIG_SNAPSHOT = NOISE_ROOT / "noise_collection_config.json"

# BoH outputs.
ASR_DATA_DIR = DATA_DIR / "asr"
BOH_DIR = ASR_DATA_DIR / "boh"
RUNTIME_BOH_PATH = ASR_DATA_DIR / "vi_boh_v1.json"
THRESHOLD_CALIBRATION_PATH = ASR_DATA_DIR / "threshold_calibration.json"

# Vietnamese speech eval set (FLEURS).
FLEURS_DIR = DATA_DIR / "fleurs_vi"
FLEURS_WAV_DIR = FLEURS_DIR / "wav"
FLEURS_MANIFEST = FLEURS_DIR / "manifest.jsonl"

# Eval outputs.
EVAL_RESULTS_DIR = REPO_ROOT / "eval" / "results"

# Model cache — gitignored.
MODELS_DIR = REPO_ROOT / "models"

# Run outputs (raw audit logs, per-run metadata).
LOCAL_OUTPUTS_DIR = REPO_ROOT / "notebooks" / "outputs"

SAMPLE_RATE = 16000
SEED = 42

# --- Noise dataset config (mirror notebooks/01) ---

TARGET_TOTAL_SAMPLES = 800

NOISE_SOURCES = {
    "esc50": {
        "enabled": True,
        "hf_repo": "ashraq/esc50",
        "split": "train",
        "n_samples": 500,
        "exclude_categories": [
            "crying_baby",
            "sneezing",
            "clapping",
            "breathing",
            "coughing",
            "footsteps",
            "laughing",
            "brushing_teeth",
            "snoring",
            "drinking_sipping",
            "crying",
            "speaking",
        ],
    },
}

SYNTHETIC_NOISE = {
    "enabled": True,
    "n_silence": 100,
    "n_white_noise": 100,
    "n_pink_noise": 100,
    "durations_s": [1.0, 3.0, 5.0, 10.0, 20.0],
    "amplitudes": [0.001, 0.003, 0.005, 0.01],
}

# --- BoH config (mirror notebooks/02) ---

MIN_COUNT = 2
MIN_CHARS = 5
NUM_THREADS = 4
MAX_NEW_TOKENS = 128

ASR_DECODE_LANG = "vi"
ASR_DECODE_TASK = "transcribe"
ASR_DECODE_STRATEGY = "greedy"

MODEL_REGISTRY = {
    key: {
        "repo_id": config.hf_repo,
        "local_subdir": config.local_dir_name,
        "params_m": config.params_m,
    }
    for key, config in ASR_MODEL_REGISTRY.items()
}

MODEL_ALLOW_PATTERNS = PHOWHISPER_ALLOW_PATTERNS

# Provider priority on Mac. CoreML first, fall back to CPU.
# Notebook 02 uses CUDA > CoreML > CPU because it may run on Colab GPU.
DEFAULT_PROVIDER_PRIORITY = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

# --- FLEURS download config ---

FLEURS_REPO = "google/fleurs"
FLEURS_LANG = "vi_vn"
FLEURS_SPLIT = "test"
FLEURS_DEFAULT_TARGET = 200  # enough for percentile p99 calibration
