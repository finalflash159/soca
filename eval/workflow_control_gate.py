from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eval.result_io import make_eval_artifact_metadata, write_json_artifact
from soca.config import SecretStore, load_settings
from soca.core.workflow import (
    ControlledWorkflowRunner,
    GoalDecision,
    GoalDecisionKind,
    GoalResolver,
    SourceKind,
    StructuredGoalResolver,
    StructuredWorkflowPlanner,
    SuccessCriterion,
)
from soca.knowledge import MarkdownVaultKnowledgeSource
from soca.llm import LocalLlamaCppLLM
from soca.llm.providers import RemoteOpenAILLM, get_provider
from soca.tools import KnowledgeReadTool, KnowledgeSearchTool, ToolRuntime

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = REPO_ROOT / "eval" / "fixtures" / "real_rag_vault"
DEFAULT_DATASET = REPO_ROOT / "eval" / "prompts" / "remediation_workflow_vi.jsonl"


def build_engine(
    backend: str,
    model: str | None,
    model_path: Path | None = None,
):
    if backend == "local":
        return LocalLlamaCppLLM(
            model_key=model or "arcee_vylinh_3b_q4_k_m",
            model_path=model_path,
            n_ctx=8_192,
            n_threads=8,
            n_gpu_layers=-1,
        )
    settings = load_settings()
    provider_key = settings.provider_key
    provider = get_provider(provider_key)
    key = SecretStore(dotenv_path=REPO_ROOT / ".env").get_key(provider_key)
    if not key:
        raise RuntimeError(f"missing API key for {provider_key}")
    return RemoteOpenAILLM(
        provider,
        model or settings.model_id,
        key,
        reasoning_enabled=settings.effective_reasoning_enabled,
        reasoning_parameter=settings.model_reasoning_parameter,
    )


def run_gate(
    *,
    backend: str,
    model: str | None,
    corpus: Path,
    dataset: Path,
    model_path: Path | None = None,
    component: str = "full",
    user_text: str = (
        "Trong ghi chú của tôi có phần nào nói về ONNX Runtime? "
        "Hãy kiểm tra đúng nguồn rồi cho tôi biết."
    ),
) -> dict[str, Any]:
    engine = build_engine(backend, model, model_path)
    source = MarkdownVaultKnowledgeSource(corpus, include_globs=("wiki/**/*.md",))
    tools = ToolRuntime(
        [
            KnowledgeSearchTool(source),
            KnowledgeReadTool(source),
        ]
    )
    goal_resolver = StructuredGoalResolver(engine, max_tokens=512)
    planner = StructuredWorkflowPlanner(
        engine,
        tools,
        max_tokens=512,
        max_actions=4,
        model_context_window=getattr(engine, "n_ctx", None),
    )
    started = time.perf_counter()
    if component == "full":
        decision = goal_resolver.decide(user_text, active_goal=None)
    else:
        decision = GoalDecision(
            kind=GoalDecisionKind.NEW,
            objective="Tìm ghi chú ONNX Runtime từ knowledge vault",
            success_criteria=(SuccessCriterion("knowledge_queried"),),
            required_sources=(SourceKind.KNOWLEDGE,),
        )
    resolution = GoalResolver().resolve(user_text, decision=decision)
    run = ControlledWorkflowRunner(tools).run(
        resolution.goal,
        planner=planner,
        initial_model_calls=decision.model_calls,
        session_id=f"workflow-{backend}",
        surface="chat",
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    public_updates = [event for event in run.events if event.event.value == "public_update"]
    executed_actions = [
        event for event in run.events if event.payload.get("operation") == "execute"
    ]
    promise_without_action = int(bool(public_updates) and not executed_actions)
    metadata = make_eval_artifact_metadata(
        suite="workflow_control_gate",
        run_type="smoke",
        data_files=(dataset, corpus / "SOURCE_MANIFEST.json"),
        config={
            "backend": backend,
            "model": model or getattr(engine, "model", None) or getattr(engine, "model_key", ""),
            "corpus": str(corpus),
            "dataset": str(dataset),
            "model_path": str(model_path) if model_path is not None else None,
            "component": component,
            "user_text": user_text,
            "max_actions": 4,
            "max_transitions": 12,
            "structured_repair_limit": 1,
        },
        ignored_untracked_paths=(Path("session.txt"),),
    )
    return {
        "schema_version": "soca-workflow-control-gate-v1",
        "artifact": metadata.to_dict(),
        "result": {
            "terminal_status": run.terminal.status.value,
            "error_code": run.terminal.error_code,
            "route": run.terminal.route,
            "goal_kind": decision.kind.value,
            "required_sources": [source.value for source in decision.required_sources],
            "planned_actions": run.budget.planned_actions,
            "tool_calls": run.budget.tool_calls,
            "model_calls": run.budget.model_calls,
            "structured_repairs": run.budget.structured_repairs,
            "public_updates": len(public_updates),
            "executed_actions": len(executed_actions),
            "promise_without_action": promise_without_action,
            "terminal_events": sum(event.terminal for event in run.events),
            "evidence_ids": list(run.terminal.evidence_ids),
            "response": run.terminal.final_text,
            "elapsed_ms": elapsed_ms,
            "goal_resolver_usage": goal_resolver.last_usage,
            "planner_prompt_manifest": planner.last_prompt_manifest,
            "planner_validation_error": planner.last_validation_error,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("local", "remote"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--component", choices=("full", "planner"), default="full")
    parser.add_argument("--query")
    parser.add_argument(
        "--expect-terminal",
        default="achieved",
        choices=(
            "achieved",
            "needs_clarification",
            "insufficient_evidence",
            "safe_failure",
            "budget_exhausted",
            "cancelled",
            "system_failure",
        ),
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_gate(
        backend=args.backend,
        model=args.model,
        corpus=args.corpus,
        dataset=args.dataset,
        model_path=args.model_path,
        component=args.component,
        user_text=args.query
        or (
            "Trong ghi chú của tôi có phần nào nói về ONNX Runtime? "
            "Hãy kiểm tra đúng nguồn rồi cho tôi biết."
        ),
    )
    write_json_artifact(args.output, report)
    print(json.dumps(report["result"], ensure_ascii=False, indent=2))
    result = report["result"]
    return int(
        result["terminal_status"] != args.expect_terminal
        or result["promise_without_action"] != 0
        or result["terminal_events"] != 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
