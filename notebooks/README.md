# SoCa Colab Notebooks

This folder contains notebooks for SoCa research, data collection, training, calibration, and export work.

The D2.5 ASR robustness notebooks can run in two modes:

- **Colab mode**: use Google Drive for heavy artifacts.
- **Local mode**: skip `google.colab` and write to ignored local folders such as `data/`, `models/`, and `notebooks/outputs/`.

Keep the boundary clear:

- Commit notebooks, small config files, and explanations.
- Do not commit downloaded datasets, generated audio, checkpoints, ONNX/GGUF exports, or benchmark outputs.
- Store heavy artifacts in Google Drive or Hugging Face, then copy them into ignored local folders only when needed.

## Recommended Notebook Order

```text
00_colab_setup.ipynb
01_noise_data_collection.ipynb
02_build_vietnamese_boh.ipynb
03_threshold_calibration.ipynb
04_table7_replication.ipynb
05_asr_finetune_phowhisper_lora.ipynb
06_export_onnx_quantize.ipynb
```

## Standard Colab Header

```python
PROJECT = "soca"
RUN_NAME = "d2_5_asr_robustness"
SEED = 42
```

```python
from pathlib import Path

GITHUB_REPO_URL = "https://github.com/finalflash159/soca.git"
REPO_DIR = Path("/content/soca")
DRIVE_ROOT = Path("/content/drive/MyDrive/soca")
```

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then run `notebooks/00_colab_setup.ipynb` to clone/pull the repo, create the Drive layout, and install D2.5 dependencies.

## Local Mac Fallback

When Colab quota is unavailable, start from the repo checkout in VS Code/Jupyter:

```text
01_noise_data_collection.ipynb
02_build_vietnamese_boh.ipynb
```

In both notebooks, keep:

```python
EXECUTION_MODE = "auto"
```

or set it explicitly:

```python
EXECUTION_MODE = "local"
```

Local mode does not import `google.colab`. It writes artifacts here:

```text
data/noise_for_boh/
models/
notebooks/outputs/{RUN_ID}/
data/asr/
```

For local CPU runs, keep `MAX_FILES` small in notebook 02 first, for example `20` to `100`.

## Artifact Layout

Use this layout on Google Drive:

```text
/content/drive/MyDrive/soca/
├── datasets/
├── checkpoints/
├── exports/
└── logs/
```

Ignored local destinations:

```text
data/
models/
eval/results/
notebooks/outputs/
notebooks/runs/
notebooks/artifacts/
```

## Notes

Notebook code should gradually move into importable Python modules or scripts once it stabilizes. Notebooks should orchestrate experiments; they should not be the only place where core logic lives.
