from __future__ import annotations

import argparse

from eval.embedding_candidates import EVAL_CANDIDATES
from soca.knowledge.retrievers.dense import default_model_home


def main() -> int:
    parser = argparse.ArgumentParser(description="Provision a Vietnamese eval embedding.")
    parser.add_argument("candidate", choices=tuple(EVAL_CANDIDATES))
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EVAL_CANDIDATES[args.candidate])
    target = default_model_home() / "eval" / args.candidate
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    model.save(str(target))
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
