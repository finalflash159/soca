from __future__ import annotations

import argparse

from eval.embedding_candidates import EVAL_CANDIDATE_FALLBACKS, EVAL_CANDIDATES
from soca.knowledge.retrievers.dense import default_model_home


def main() -> int:
    parser = argparse.ArgumentParser(description="Provision a Vietnamese eval embedding.")
    parser.add_argument("candidate", choices=tuple(EVAL_CANDIDATES))
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="load only from the local Hugging Face cache; never contact the network",
    )
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    source_ids = (
        EVAL_CANDIDATES[args.candidate],
        *EVAL_CANDIDATE_FALLBACKS.get(args.candidate, ()),
    )
    model = None
    load_errors: list[tuple[str, OSError]] = []
    for source_id in source_ids:
        try:
            model = SentenceTransformer(
                source_id,
                local_files_only=args.local_files_only,
            )
            break
        except OSError as exc:
            load_errors.append((source_id, exc))
    if model is None:
        details = "; ".join(f"{source}: {error}" for source, error in load_errors)
        raise OSError(f"could not load {args.candidate} from any registered source: {details}")
    target = default_model_home() / "eval" / args.candidate
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    model.save(str(target))
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
