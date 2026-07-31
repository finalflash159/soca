from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NLTK_DATA = REPO_ROOT / "models" / "nltk_data"
# nltk >=3.8.2 renamed the POS tagger package; g2p_en needs the tagger, the
# legacy/renamed pair covers both nltk generations without probing versions.
PACKAGES = ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "cmudict")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Provision nltk data g2p_en needs for OOV English G2P."
    )
    parser.add_argument("--nltk-data", type=Path, default=DEFAULT_NLTK_DATA)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.nltk_data.mkdir(parents=True, exist_ok=True)

    # nltk's download-path sandbox (pathsec.py) only allows writing into
    # directories already on NLTK_DATA/nltk.data.path, so this must be set
    # before nltk is imported.
    os.environ["NLTK_DATA"] = str(args.nltk_data)

    import nltk

    for package in PACKAGES:
        nltk.download(package, download_dir=str(args.nltk_data))
    print(args.nltk_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
