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
});
