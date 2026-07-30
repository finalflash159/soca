"""Opt-in live LLM router evaluation without executing selected tools."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from eval.eval_tool_router import _load, _query, evaluate
from soca.config import LlmSettings, SecretStore
from soca.core.llm_tool_router import LLMToolRouter
from soca.core.tool_routing import ToolRouterConfig
from soca.llm.factory import build_llm_engine
from soca.tools import ToolRuntime, ToolSpec, object_schema


class _CatalogTool:
    def __init__(self, spec: ToolSpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def run(self, arguments: dict[str, Any]):  # pragma: no cover - never executed by this command
        raise RuntimeError("live eval never executes tools")


def _catalog() -> ToolRuntime:
    return ToolRuntime(
        [
            _CatalogTool(
                ToolSpec(
                    "knowledge.inspect",
                    "Inspect bounded local vault navigation metadata and explicit links; this is not content evidence.",
                    object_schema(
                        properties={
                            "scope": {"type": "string"},
                            "path": {"type": "string"},
                            "depth": {"type": "integer"},
                            "limit": {"type": "integer"},
                        }
                    ),
                )
            ),
            _CatalogTool(
                ToolSpec(
                    "knowledge.search",
                    "Search local wiki notes.",
                    object_schema(
                        properties={
                            "query": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                        required=["query"],
                    ),
                )
            ),
            _CatalogTool(
                ToolSpec(
                    "knowledge.read",
                    "Read one scoped wiki markdown path.",
                    object_schema(
                        properties={"path": {"type": "string"}},
                        required=["path"],
                    ),
                )
            ),
            _CatalogTool(
                ToolSpec(
                    "memory.search",
                    "Search private local memory notes.",
                    object_schema(
                        properties={
                            "query": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                        required=["query"],
                    ),
                )
            ),
        ]
    )


def run_live(
    dataset: Path,
    output: Path,
    *,
    provider: str,
    model: str,
    max_cases: int,
) -> dict[str, Any]:
    rows = _load(dataset)[:max_cases]
    # Provider settings enforce the product-wide minimum output budget.  The
    # router's own request remains bounded below; it is passed separately in
    # ToolRouterConfig rather than weakening LlmSettings validation.
    settings = LlmSettings(backend="remote", provider_key=provider, model_id=model, max_tokens=2048)
    llm = build_llm_engine(settings, SecretStore())
    runtime = _catalog()
    router = LLMToolRouter(
        llm,
        runtime,
        config=ToolRouterConfig(mode="llm", max_tokens=96, repair_attempts=1),
    )
    predictions: list[dict[str, Any]] = []
    for row in rows:
        started = time.perf_counter()
        call = router.select(_query(row), knowledge_limit=3)
        predictions.append(
            {
                "id": row["id"],
                "tool": call.name if call is not None else "none",
                "tier": router.last_tier,
                "reason": getattr(router.last_decision, "reason", "llm_none"),
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions))
    return evaluate(dataset, output, tier="llm")


def main() -> int:
    parser = argparse.ArgumentParser(description="Opt-in OpenRouter live router eval; incurs provider cost.")
    parser.add_argument("--provider", choices=["openrouter"], default="openrouter")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=25, help="Safety cap; increase explicitly for a full run.")
    args = parser.parse_args()
    if args.max_cases < 1:
        parser.error("--max-cases must be positive")
    result = run_live(
        args.dataset,
        args.predictions,
        provider=args.provider,
        model=args.model,
        max_cases=args.max_cases,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
