import { describe, expect, it } from "vitest";
import type { TurnProgressEvent } from "./protocol.js";
import { initialState, reduce } from "./store.js";

function progress(
  phase: TurnProgressEvent["phase"],
  operation: string = phase,
): TurnProgressEvent {
  return {
    event: "turn_progress",
    surface: "chat",
    phase,
    operation,
    status: "active",
  };
}

describe("progress reducer", () => {
  it("queues rapid real stages instead of overwriting them with synthesis", () => {
    let state = reduce(initialState, {
      type: "engine_event",
      event: progress("analyzing"),
    });
    state = reduce(state, {
      type: "engine_event",
      event: progress("routing"),
    });
    state = reduce(state, {
      type: "engine_event",
      event: progress("memory"),
    });
    state = reduce(state, {
      type: "engine_event",
      event: progress("retrieval", "tool:knowledge.search"),
    });
    state = reduce(state, {
      type: "engine_event",
      event: progress("synthesis", "llm"),
    });

    expect(state.turnProgress?.phase).toBe("analyzing");
    expect(state.progressQueue.map((event) => event.phase)).toEqual([
      "routing",
      "memory",
      "retrieval",
      "synthesis",
    ]);

    state = reduce(state, { type: "advance_progress" });
    expect(state.turnProgress?.phase).toBe("routing");
    expect(state.progressQueue.map((event) => event.phase)).toEqual([
      "memory",
      "retrieval",
      "synthesis",
    ]);
  });

  it("deduplicates adjacent phases and clears the queue when done", () => {
    let state = reduce(initialState, {
      type: "engine_event",
      event: progress("analyzing", "normalize_input"),
    });
    state = reduce(state, {
      type: "engine_event",
      event: progress("analyzing", "input_guardrail"),
    });
    expect(state.turnProgress?.operation).toBe("input_guardrail");
    expect(state.progressQueue).toEqual([]);

    state = reduce(state, {
      type: "engine_event",
      event: { ...progress("complete"), status: "done" },
    });
    expect(state.turnProgress).toBeNull();
    expect(state.progressQueue).toEqual([]);
  });

  it("ignores stale progress from an older run and retains a failed terminal", () => {
    const event = (
      run_id: string,
      sequence: number,
      status: TurnProgressEvent["status"],
      phase: TurnProgressEvent["phase"],
    ): TurnProgressEvent => ({
      event: "turn_progress",
      surface: "chat",
      phase,
      operation: phase,
      status,
      run_id,
      goal_id: `goal-${run_id}`,
      sequence,
      terminal_status: status === "failed" ? "system_failure" : undefined,
    });

    let state = reduce(initialState, {
      type: "engine_event",
      event: event("run-old", 3, "active", "retrieval"),
    });
    state = reduce(state, {
      type: "engine_event",
      event: event("run-new", 0, "active", "analyzing"),
    });
    state = reduce(state, {
      type: "engine_event",
      event: event("run-old", 4, "active", "synthesis"),
    });
    expect(state.turnProgress?.run_id).toBe("run-new");
    expect(state.turnProgress?.phase).toBe("analyzing");

    state = reduce(state, {
      type: "engine_event",
      event: event("run-new", 1, "failed", "complete"),
    });
    expect(state.turnProgress?.status).toBe("failed");
    expect(state.progressQueue).toEqual([]);
  });

  it("keeps provisional workflow text separate until the chat terminal arrives", () => {
    let state = reduce(initialState, {
      type: "engine_event",
      event: {
        event: "answer_delta",
        protocol_version: 2,
        session_id: "session",
        run_id: "run",
        goal_id: "goal",
        sequence: 1,
        surface: "chat",
        timestamp: "2026-07-30T00:00:00Z",
        node: "synthesize",
        status: "active",
        payload: { text: "đang tổng hợp" },
      },
    });
    expect(state.pendingAnswer).toBe("đang tổng hợp");
    state = reduce(state, {
      type: "engine_event",
      event: {
        event: "turn_terminal",
        protocol_version: 2,
        session_id: "session",
        run_id: "run",
        goal_id: "goal",
        sequence: 2,
        surface: "chat",
        timestamp: "2026-07-30T00:00:01Z",
        node: "finalize",
        status: "completed",
        payload: { terminal_status: "achieved", final_text: "xong" },
      },
    });
    expect(state.pendingAnswer).toBe("");
    expect(state.workflowEvents).toHaveLength(2);
  });

  it("does not duplicate voice draft text mirrored by workflow events", () => {
    let state = reduce(initialState, {
      type: "engine_event",
      event: {
        event: "voice",
        type: "llm_token",
        text: "xin chào",
        latency_ms: null,
        metadata: {},
        usage: null,
      },
    });
    state = reduce(state, {
      type: "engine_event",
      event: {
        event: "answer_delta",
        protocol_version: 2,
        session_id: "session",
        run_id: "voice-run",
        goal_id: "voice-goal",
        sequence: 1,
        surface: "voice",
        timestamp: "2026-07-30T00:00:00Z",
        node: "synthesize",
        status: "active",
        payload: { text: "xin chào" },
      },
    });

    expect(state.pendingAnswer).toBe("xin chào");
  });

  it("keeps citations as protocol data and closes temporary info on a new turn", () => {
    let state = reduce(
      { ...initialState, activeInfo: "status" },
      { type: "user_message", text: "attention" },
    );
    expect(state.activeInfo).toBeNull();

    state = reduce(state, {
      type: "engine_event",
      event: {
        event: "chat",
        type: "done",
        text: "Attention dùng query, key và value.",
        citations: [
          {
            label: "K1",
            path: "wiki/learning/attention.md",
            title: "Attention",
            line_start: 12,
            line_end: 18,
            source: "knowledge",
          },
        ],
      },
    });

    expect(state.timeline.at(-1)).toMatchObject({
      kind: "soca",
      text: "Attention dùng query, key và value.",
      citations: [
        {
          label: "K1",
          path: "wiki/learning/attention.md",
          source: "knowledge",
        },
      ],
    });
  });
});
