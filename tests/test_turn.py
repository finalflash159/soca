from soca.core import RuntimeResult, RuntimeRoute, RuntimeTrace, TurnFrame


def test_turn_frame_defaults_to_text_source() -> None:
    frame = TurnFrame("xin chào")

    assert frame.text == "xin chào"
    assert frame.source == "text"
    assert frame.metadata == {}


def test_runtime_trace_captures_route_and_flags() -> None:
    trace = RuntimeTrace(
        route=RuntimeRoute.LLM_FALLBACK,
        used_llm=True,
        stage_latencies_ms={"llm": 12.5},
    )

    assert trace.route == RuntimeRoute.LLM_FALLBACK
    assert trace.used_llm is True
    assert trace.used_tool is False
    assert trace.blocked is False
    assert trace.stage_latencies_ms["llm"] == 12.5


def test_runtime_result_links_trace_and_frame() -> None:
    frame = TurnFrame("mấy giờ rồi", source="asr")
    trace = RuntimeTrace(route=RuntimeRoute.TOOL_DIRECT, used_tool=True)
    result = RuntimeResult(
        response_text="Bây giờ là 09:00.",
        route=RuntimeRoute.TOOL_DIRECT,
        frame=frame,
        trace=trace,
    )

    assert result.blocked is False
    assert result.frame is frame
    assert result.trace is trace
    assert result.route == RuntimeRoute.TOOL_DIRECT

