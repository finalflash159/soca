from __future__ import annotations

import argparse

from soca.knowledge.retrievers.dense import FastEmbedModel, Model2VecModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Warm a SoCa embedding model cache.")
    parser.add_argument("backend", choices=("fastembed", "model2vec"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model = (
        FastEmbedModel(allow_download=True)
        if args.backend == "fastembed"
        else Model2VecModel(allow_download=True)
    )
    vector = model.embed_query("kiểm tra mô hình tìm kiếm")
    print(f"{model.model_id}: dimension={vector.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
