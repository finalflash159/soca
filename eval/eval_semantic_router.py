"""Score semantic-router captures with the same v2 metrics as the cascade."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from eval.eval_tool_router import evaluate as evaluate_router
from soca.core.semantic_tool_router import build_semantic_router
from soca.core.tool_routing import SemanticRouterConfig
from soca.knowledge.retrievers.dense import FastEmbedModel
from soca.tools import ToolRuntime, ToolSpec, object_schema


class _CatalogTool:
    def __init__(self, spec: ToolSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def run(self, arguments: dict[str, Any]):  # pragma: no cover - eval never executes tools
        raise RuntimeError("semantic eval never executes tools")


def _catalog() -> ToolRuntime:
    search_schema = object_schema(
        properties={"query": {"type": "string"}, "limit": {"type": "integer"}},
        required=["query"],
    )
    return ToolRuntime(
        [
            _CatalogTool(ToolSpec("local_time.now", "Return local time.", object_schema())),
            _CatalogTool(ToolSpec("knowledge.search", "Search wiki.", search_schema)),
            _CatalogTool(ToolSpec("memory.search", "Search private memory.", search_schema)),
        ]
    )


def run_local(
    dataset: Path,
    examples: Path,
    predictions: Path,
    *,
    threshold: float = 0.58,
    margin: float = 0.04,
) -> dict[str, Any]:
    from eval.eval_tool_router import _load, _query

    model = FastEmbedModel(allow_download=False)
    router = build_semantic_router(
        tool_runtime=_catalog(),
        config=SemanticRouterConfig(
            enabled=True,
            threshold=threshold,
            margin=margin,
            examples_path=examples,
        ),
        embedding_model=model,
    )
    if router is None:
        raise RuntimeError("semantic router could not be constructed")
    rows = _load(dataset)
    records: list[dict[str, Any]] = []
    for row in rows:
        started = time.perf_counter()
        call = router.select(_query(row), knowledge_limit=3)
        records.append(
            {
                "id": row["id"],
                "tool": call.name if call is not None else "none",
                "tier": router.last_tier,
                "reason": router.last_decision.reason,
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
        )
    predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return evaluate_router(dataset, predictions, tier="semantic")


def evaluate(dataset: Path, predictions: Path | None = None) -> dict[str, Any]:
    if predictions is None:
        legacy_rows = []
        with dataset.open(encoding="utf-8") as handle:
            legacy_rows = [json.loads(line) for line in handle if line.strip()]
        if legacy_rows and not any(row.get("query") or row.get("text") for row in legacy_rows):
            return {"case_count": len(legacy_rows), "coverage": 0.0, "accuracy": 0.0}
    return evaluate_router(dataset, predictions, tier="semantic")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path, required=True, help="Recorded examples file for audit metadata.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--run-local", action="store_true", help="Run the provisioned local embedder and capture predictions.")
    parser.add_argument("--threshold", type=float, default=0.58)
    parser.add_argument("--margin", type=float, default=0.04)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.run_local:
        if args.predictions is None:
            parser.error("--predictions is required with --run-local")
        result = run_local(
            args.dataset,
            args.examples,
            args.predictions,
            threshold=args.threshold,
            margin=args.margin,
        )
    else:
        result = evaluate(args.dataset, args.predictions)
    result["examples"] = str(args.examples)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
