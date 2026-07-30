from __future__ import annotations

from dataclasses import dataclass

from eval.baseline_cases import RemediationCase
from eval.runtime_remediation_baseline import (
    BaselineRunConfig,
    build_runtime_report,
    run_runtime_baseline,
)
from soca.core import (
    LLMUsage,
    RuntimeResult,
    RuntimeRoute,
    RuntimeStreamEvent,
    RuntimeTrace,
)
from soca.knowledge import KnowledgeCitation
from soca.tools import ToolCall, ToolResult


@dataclass
class ScriptedBaselineRuntime:
    streaming: bool = False

    def _result(self) -> RuntimeResult:
        citation = KnowledgeCitation(
            path="wiki/xquad_vi/Oxygen.md",
            title="Oxygen",
        )
        trace = RuntimeTrace(
            route=RuntimeRoute.KNOWLEDGE_LLM,
            tool_calls=(ToolCall("knowledge.search", {"query": "oxygen"}),),
            tool_results=(ToolResult("knowledge.search", True, "evidence"),),
            citations=(citation,),
            used_tool=True,
            used_llm=True,
            selected_sources=("knowledge",),
            evidence_status="supported",
            stage_latencies_ms={"tool_router": 1.0, "llm": 2.0},
        )
        return RuntimeResult(
            response_text="Khí hậu cổ [K1].",
            route=RuntimeRoute.KNOWLEDGE_LLM,
            citations=(citation,),
            trace=trace,
            usage=LLMUsage(
                prompt_tokens=100,
                completion_tokens=10,
                ttft_ms=20,
                total_latency_ms=50,
                tokens_per_second=200,
            ),
        )

    def run_text_turn(self, text, *, source="text", metadata=None):
        return self._result()

    def stream_text_turn(self, text, *, source="text", metadata=None):
        yield RuntimeStreamEvent("token", text="Khí")
        yield RuntimeStreamEvent("result", result=self._result())


def _case() -> RemediationCase:
    return RemediationCase(
        case_id="runtime-contract",
        suite_kind="capability",
        dataset_class="public_screening",
        split="test",
        family="oxygen-runtime-contract",
        category="answerable",
        turns=("oxygen?",),
        expected_goal="answer from corpus",
        expected_terminal="achieved",
        expected_sources=("knowledge",),
        expected_tools=("knowledge.search",),
        expected_citations=("wiki/xquad_vi/Oxygen.md",),
        audit_items=("P0-1",),
        provenance="contract fixture",
        metadata={},
    )


def test_runtime_baseline_records_trace_usage_and_terminal() -> None:
    records = run_runtime_baseline(
        (_case(),),
        runtime=ScriptedBaselineRuntime(),
    )

    assert records[0]["outcome"]["passed"] is True
    turn = records[0]["turns"][0]
    assert turn["route"] == "knowledge_llm"
    assert turn["tool_calls"][0]["name"] == "knowledge.search"
    assert turn["citations"][0]["path"] == "wiki/xquad_vi/Oxygen.md"
    assert turn["usage"]["prompt_tokens"] == 100
    assert turn["latency_ms"]["wall"] >= 0


def test_streaming_baseline_requires_and_records_terminal_result() -> None:
    records = run_runtime_baseline(
        (_case(),),
        runtime=ScriptedBaselineRuntime(streaming=True),
        execution_mode="streaming",
    )

    assert records[0]["error"] is None
    assert records[0]["turns"][0]["legacy_terminal"]["status"] == "succeeded"


def test_report_keeps_capability_and_regression_counts_separate() -> None:
    records = run_runtime_baseline(
        (_case(),),
        runtime=ScriptedBaselineRuntime(),
    )
    report = build_runtime_report(
        records=records,
        metadata={"source": {"commit": "abc"}},
        run_config=BaselineRunConfig(
            execution_mode="blocking",
            provider="test",
            model="scripted",
            backend="test",
            retrieval_mode="test",
            dense_backend="test",
            max_tokens=4096,
        ),
    )

    assert report["summary"]["by_suite"]["capability"] == {
        "cases": 1,
        "passed": 1,
        "failed": 0,
    }
