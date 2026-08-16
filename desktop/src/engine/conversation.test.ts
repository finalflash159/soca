import { describe, expect, it } from "vitest";

import {
  initialConversation,
  reduceConversation,
  turnStatus,
  turnText,
} from "./conversation";
import type { EngineFrame } from "./protocol";
import { PROTOCOL_VERSION } from "./protocol";

const RUN = "run-1";

function fold(frames: EngineFrame[]) {
  return frames.reduce(reduceConversation, initialConversation);
}

const start = (text: string, runId = RUN): EngineFrame =>
  ({ event: "chat", type: "start", text, run_id: runId, goal_id: "g" }) as EngineFrame;

const delta = (text: string, runId = RUN): EngineFrame =>
  ({
    event: "answer_delta",
    protocol_version: PROTOCOL_VERSION,
    session_id: "s",
    run_id: runId,
    goal_id: "g",
    sequence: 1,
    surface: "chat",
    timestamp: "2026-08-15T00:00:00Z",
    node: "synthesize",
    status: "active",
    payload: { text },
  }) as EngineFrame;

const done = (text: string, extra: Record<string, unknown> = {}): EngineFrame =>
  ({ event: "chat", type: "done", text, route: "smalltalk", blocked: false, ...extra }) as EngineFrame;

const terminal = (status: string, runId = RUN): EngineFrame =>
  ({
    event: "turn_terminal",
    protocol_version: PROTOCOL_VERSION,
    session_id: "s",
    run_id: runId,
    goal_id: "g",
    sequence: 9,
    surface: "chat",
    timestamp: "2026-08-15T00:00:00Z",
    node: "finalize",
    status: "completed",
    payload: { status },
  }) as EngineFrame;

describe("streaming assembly", () => {
  it("appends deltas rather than replacing them", () => {
    const state = fold([start("hỏi gì đó"), delta("Cơ bắp "), delta("tạo lực."), done("Cơ bắp tạo lực.")]);
    expect(state.turns[0].streamedText).toBe("Cơ bắp tạo lực.");
    expect(state.turns[0].deltaCount).toBe(2);
  });

  it("preserves chunk-boundary whitespace exactly", () => {
    // Regression shape for the caption bug: per-chunk cleaning that strips
    // leading/trailing space reassembles as "cơ bắp.Nó tạo".
    const state = fold([
      start("q"),
      delta("Nó làm khỏe cơ bắp. "),
      delta("Nó tạo lực."),
      done("Nó làm khỏe cơ bắp. Nó tạo lực."),
    ]);
    expect(state.turns[0].streamedText).toBe("Nó làm khỏe cơ bắp. Nó tạo lực.");
    expect(state.reassemblyMismatch).toBe(false);
  });

  it("flags a stream that does not reassemble into the final answer", () => {
    const state = fold([start("q"), delta("Cơ bắp."), delta("Nó tạo"), done("Cơ bắp. Nó tạo")]);
    expect(state.reassemblyMismatch).toBe(true);
  });

  it("does not flag trailing whitespace, which the cleaners differ on by design", () => {
    // answer_chunk_* preserves chunk edges so words do not glue together;
    // answer_text_* ends with .strip(). Comparing exactly would fire on almost
    // every answer (docs/18 §6).
    const state = fold([
      start("q"),
      delta("Xin chào. "),
      delta("Bạn khỏe không?\n"),
      done("Xin chào. Bạn khỏe không?"),
    ]);
    expect(state.reassemblyMismatch).toBe(false);
  });

  it("still flags an interior divergence", () => {
    const state = fold([start("q"), delta("Cơ bắp."), delta("Nó tạo"), done("Cơ bắp. Nó tạo")]);
    expect(state.reassemblyMismatch).toBe(true);
  });

  it("does not flag a turn that produced no deltas at all", () => {
    // A tool turn can publish everything through chat/done.
    const state = fold([start("q"), done("câu trả lời")]);
    expect(state.reassemblyMismatch).toBe(false);
  });

  it("treats a single delta as normal", () => {
    const state = fold([start("q"), delta("toàn bộ câu trả lời"), done("toàn bộ câu trả lời")]);
    expect(state.turns[0].deltaCount).toBe(1);
    expect(state.reassemblyMismatch).toBe(false);
  });

  it("ignores empty deltas", () => {
    const state = fold([start("q"), delta(""), delta("xin chào")]);
    expect(state.turns[0].deltaCount).toBe(1);
    expect(state.turns[0].streamedText).toBe("xin chào");
  });
});

describe("turn text", () => {
  it("shows the streamed prefix while the turn is open", () => {
    const state = fold([start("q"), delta("một nửa")]);
    expect(turnText(state.turns[0])).toBe("một nửa");
    expect(turnStatus(state.turns[0])).toBe("streaming");
  });

  it("prefers the final answer once it arrives", () => {
    const state = fold([start("q"), delta("một nửa"), done("một nửa và phần còn lại")]);
    expect(turnText(state.turns[0])).toBe("một nửa và phần còn lại");
  });
});

describe("terminal outcomes", () => {
  it("marks blocked as its own status, not as a failure", () => {
    const state = fold([
      start("q"),
      done("", { blocked: true }),
      terminal("insufficient_evidence"),
    ]);
    expect(turnStatus(state.turns[0])).toBe("blocked");
    expect(state.turns[0].terminal).toBe("insufficient_evidence");
  });

  it("marks chat:error as failed", () => {
    const state = fold([start("q"), { event: "chat", type: "error", text: "boom" } as EngineFrame]);
    expect(turnStatus(state.turns[0])).toBe("failed");
    expect(state.turns[0].error).toBe("boom");
  });

  it("records citations from chat:done", () => {
    const state = fold([start("q"), done("có nguồn", { citations: [{ label: "K1" }] })]);
    expect(state.turns[0].citations).toHaveLength(1);
  });
});

describe("surface routing", () => {
  it("ignores voice deltas, which are raw tokens with a separate caption", () => {
    // docs/18 §6: only chat deltas reassemble into chat/done.
    const voiceDelta = {
      ...(delta("token") as Record<string, unknown>),
      surface: "voice",
    } as unknown as EngineFrame;
    const state = fold([start("q"), voiceDelta]);
    expect(state.turns[0].streamedText).toBe("");
    expect(state.turns[0].deltaCount).toBe(0);
  });
});

describe("multiple turns", () => {
  it("routes deltas to the turn matching run_id", () => {
    const state = fold([
      start("first", "run-a"),
      done("first answer"),
      start("second", "run-b"),
      delta("second ", "run-b"),
      delta("answer", "run-b"),
    ]);
    expect(state.turns).toHaveLength(2);
    expect(state.turns[0].finalText).toBe("first answer");
    expect(state.turns[1].streamedText).toBe("second answer");
  });

  it("closes only the open turn on chat:done", () => {
    const state = fold([start("first", "run-a"), done("A"), start("second", "run-b"), done("B")]);
    expect(state.turns.map((turn) => turn.finalText)).toEqual(["A", "B"]);
  });

  it("drops a delta that belongs to no known turn", () => {
    expect(fold([delta("orphan")]).turns).toHaveLength(0);
  });
});

describe("phase tracking", () => {
  it("follows turn_progress for the open turn", () => {
    const state = fold([
      start("q"),
      { event: "turn_progress", run_id: RUN, phase: "retrieval" } as EngineFrame,
    ]);
    expect(state.turns[0].phase).toBe("retrieval");
  });

  it("clears the phase once the turn closes", () => {
    const state = fold([
      start("q"),
      { event: "turn_progress", run_id: RUN, phase: "synthesis" } as EngineFrame,
      done("xong"),
    ]);
    expect(state.turns[0].phase).toBeNull();
  });
});

describe("unknown frames", () => {
  it("leaves the conversation untouched", () => {
    expect(fold([{ event: "usage", turns: 2 } as EngineFrame])).toEqual(initialConversation);
  });
});
