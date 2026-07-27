from huggingface_hub import snapshot_download
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Download
local = snapshot_download("richiejp/aec-challenge-16k", local_dir=REPO_ROOT / "data" / "aec", repo_type="dataset")

# Extract all shards
for tar_path in sorted(Path(local).rglob("*.tar")):
    with tarfile.open(tar_path) as tf:
        tf.extractall(tar_path.parent)
