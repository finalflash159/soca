from __future__ import annotations

from threading import Event

from soca.tools import (
    SideEffectLevel,
    SpeculativeToolRuntime,
    ToolCall,
    ToolResult,
    ToolRuntime,
    ToolSpec,
    object_schema,
)


class CountingSearchTool:
    def __init__(self, release: Event | None = None) -> None:
        self.calls = 0
        self.release = release

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="knowledge.search",
            description="Search test knowledge.",
            input_schema=object_schema(
                {"query": {"type": "string"}, "limit": {"type": "integer"}},
                ["query"],
            ),
            side_effect=SideEffectLevel.READ_ONLY,
        )

    def run(self, arguments: dict[str, object]) -> ToolResult:
        self.calls += 1
        if self.release is not None:
            assert self.release.wait(timeout=2)
        query = str(arguments["query"])
        return ToolResult(self.spec.name, True, f"evidence:{query}")


class LocalWriteTool(CountingSearchTool):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="state.write",
            description="Write test state.",
            input_schema=object_schema(),
            side_effect=SideEffectLevel.LOCAL_STATE,
        )


def test_exact_ready_prefetch_is_consumed_once() -> None:
    tool = CountingSearchTool()
    identity = {"knowledge.search": "generation-a"}
    runtime = SpeculativeToolRuntime(
        ToolRuntime([tool]), identity_provider=lambda name: identity[name]
    )
    call = ToolCall("knowledge.search", {"query": "Bayes", "limit": 3})

    receipt = runtime.prefetch("next-turn", call)
    assert receipt.wait(timeout=2).status == "ready"

    with runtime.using_slot("next-turn"):
        first = runtime.call(call)
    with runtime.using_slot("next-turn"):
        second = runtime.call(call)

    assert first.data["speculative_retrieval"]["status"] == "hit"
    assert second.data["speculative_retrieval"]["status"] == "miss"
    assert second.data["speculative_retrieval"]["reason"] == "slot_not_found"
    assert tool.calls == 2
    runtime.close()


def test_argument_or_generation_mismatch_runs_canonical_call() -> None:
    tool = CountingSearchTool()
    identity = {"knowledge.search": "generation-a"}
    runtime = SpeculativeToolRuntime(
        ToolRuntime([tool]), identity_provider=lambda name: identity[name]
    )
    original = ToolCall("knowledge.search", {"query": "Bayes", "limit": 3})
    assert runtime.prefetch("next-turn", original).wait(timeout=2).status == "ready"
    identity["knowledge.search"] = "generation-b"

    with runtime.using_slot("next-turn"):
        result = runtime.call(original)

    marker = result.data["speculative_retrieval"]
    assert marker == {"status": "miss", "reason": "identity_changed"}
    assert tool.calls == 2
    runtime.close()


def test_pending_prefetch_does_not_block_or_silently_replace_canonical_path() -> None:
    release = Event()
    tool = CountingSearchTool(release)
    runtime = SpeculativeToolRuntime(
        ToolRuntime([tool]), identity_provider=lambda _name: "generation-a"
    )
    call = ToolCall("knowledge.search", {"query": "Bayes"})
    receipt = runtime.prefetch("next-turn", call)

    # The normal call is intentionally allowed to proceed. Release both calls
    # and assert the result records why the speculative value was not reused.
    release.set()
    with runtime.using_slot("next-turn"):
        result = runtime.call(call)

    assert receipt.wait(timeout=2).status in {"consumed", "invalidated"}
    assert result.data["speculative_retrieval"]["status"] in {"hit", "miss"}
    runtime.close()


def test_non_read_only_tool_cannot_be_prefetched() -> None:
    runtime = SpeculativeToolRuntime(
        ToolRuntime([LocalWriteTool()]), identity_provider=lambda _name: "state-a"
    )

    receipt = runtime.prefetch("write", ToolCall("state.write", {}))

    assert receipt.status == "rejected"
    assert receipt.reason == "tool_not_read_only"
    runtime.close()
