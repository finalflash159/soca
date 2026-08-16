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
    const state = fold([start("hỏi gì đó"), delta("Cơ bắp"), delta("tạo lực."), done("Cơ bắp tạo lực.")]);
    expect(state.turns[0].streamedText).toBe("Cơ bắp tạo lực.");
    expect(state.turns[0].deltaCount).toBe(2);
  });

  it("does not double the separator when a chunk already carries one", () => {
    // Voice chunks keep their edges; only stripped chat sentences need a
    // separator inserted.
    const state = fold([
      start("q"),
      delta("Nó làm khỏe cơ bắp. "),
      delta("Nó tạo lực."),
      done("Nó làm khỏe cơ bắp. Nó tạo lực."),
    ]);
    expect(state.turns[0].streamedText).toBe("Nó làm khỏe cơ bắp. Nó tạo lực.");
    expect(state.reassemblyMismatch).toBe(false);
  });

  it("joins stripped sentences with a separator", () => {
    // pop_ready_sentence returns buffer[:end].strip() and lstrips the rest, so
    // the space between two sentences reaches no client. Concatenating gives
    // "Sơn Ca.Hôm nay"; joining with a separator gives the answer.
    const state = fold([
      start("q"),
      delta("Xin chào! Mình là Sơn Ca."),
      delta("Hôm nay mình giúp gì được?"),
      done("Xin chào! Mình là Sơn Ca. Hôm nay mình giúp gì được?"),
    ]);
    expect(state.turns[0].streamedText).toBe(
      "Xin chào! Mình là Sơn Ca. Hôm nay mình giúp gì được?",
    );
    expect(state.reassemblyMismatch).toBe(false);
  });

  it("flags a dropped frame", () => {
    const state = fold([
      start("q"),
      delta("Câu một."),
      delta("Câu ba."),
      done("Câu một. Câu hai. Câu ba."),
    ]);
    expect(state.reassemblyMismatch).toBe(true);
  });

  it("does not flag trailing whitespace, which the cleaners differ on by design", () => {
    // answer_chunk_* preserves chunk edges so words do not glue together;
    // answer_text_* ends with .strip(). Comparing exactly would fire on almost
    // every answer (docs/18 §6).
    const state = fold([
      start("q"),
      delta("Xin chào."),
      delta("Bạn khỏe không?"),
      done("Xin chào. Bạn khỏe không?\n"),
    ]);
    expect(state.reassemblyMismatch).toBe(false);
  });

  it("flags a footer only the whole-answer cleaner removes", () => {
    // The model is told not to write one; when it does, the stream shows text
    // the final answer hides, and that is worth surfacing.
    const state = fold([
      start("q"),
      delta("Câu trả lời."),
      delta("Nguồn: wiki/a.md"),
      done("Câu trả lời."),
    ]);
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

const voice = (
  type: string,
  text = "",
  metadata: Record<string, unknown> = {},
): EngineFrame => ({ event: "voice", type, text, metadata }) as EngineFrame;

describe("voice turns", () => {
  it("builds a spoken turn from asr, sentence and done", () => {
    // The regression this exists for: voice reduced only into live signals, so
    // a finished spoken turn left no history behind at all.
    const state = fold([
      voice("asr", "Thời tiết hôm nay thế nào?"),
      voice("sentence", "Hôm nay trời nắng."),
      voice("sentence", "Nhiệt độ khoảng 30 độ."),
      voice("done", "Hôm nay trời nắng. Nhiệt độ khoảng 30 độ.", {
        terminal_status: "achieved",
      }),
    ]);
    expect(state.turns).toHaveLength(1);
    expect(state.turns[0].surface).toBe("voice");
    expect(state.turns[0].userText).toBe("Thời tiết hôm nay thế nào?");
    expect(turnText(state.turns[0])).toBe("Hôm nay trời nắng. Nhiệt độ khoảng 30 độ.");
    expect(turnStatus(state.turns[0])).toBe("achieved");
  });

  it("shows sentences before done lands", () => {
    const state = fold([voice("asr", "Chào bạn"), voice("sentence", "Chào bạn!")]);
    expect(turnText(state.turns[0])).toBe("Chào bạn!");
    expect(turnStatus(state.turns[0])).toBe("streaming");
  });

  it("separates sentences the splitter stripped", () => {
    // Same rule as chat: pop_ready_sentence discards the space on both sides.
    const state = fold([
      voice("asr", "hỏi"),
      voice("sentence", "Câu một."),
      voice("sentence", "Câu hai."),
    ]);
    expect(turnText(state.turns[0])).toBe("Câu một. Câu hai.");
  });

  it("ignores an empty transcript rather than showing a blank bubble", () => {
    expect(fold([voice("asr", "   ")]).turns).toHaveLength(0);
  });

  it("renders a repair as a turn, not a failure", () => {
    // docs/18 §5: rejected speech becomes a question, never an invented
    // transcript. Styling it as an error would be wrong.
    const state = fold([
      voice("asr", ""),
      voice("repair", "Bạn nói lại giúp mình nhé?"),
      voice("done", "", { rejected: true, terminal_status: "needs_clarification" }),
    ]);
    expect(state.turns).toHaveLength(1);
    expect(state.turns[0].repair).toBe("Bạn nói lại giúp mình nhé?");
    expect(state.turns[0].error).toBeNull();
  });

  it("keeps the repair prompt when done carries no text", () => {
    const state = fold([
      voice("repair", "Mình chưa nghe rõ."),
      voice("done", "", { rejected: true }),
    ]);
    expect(state.turns[0].repair).toBe("Mình chưa nghe rõ.");
  });

  it("marks a barge-in answer interrupted instead of dropping it", () => {
    const state = fold([
      voice("asr", "kể chuyện đi"),
      voice("sentence", "Ngày xửa ngày xưa."),
      voice("interrupted"),
      voice("done", "Ngày xửa ngày xưa.", { terminal_status: "cancelled" }),
    ]);
    expect(state.turns[0].interrupted).toBe(true);
    expect(turnText(state.turns[0])).toBe("Ngày xửa ngày xưa.");
  });

  it("closes an open turn when the loop stops mid-answer", () => {
    // Otherwise the bubble spins forever after the user turns voice off.
    const state = fold([
      voice("asr", "câu hỏi"),
      voice("sentence", "Đang trả lời"),
      voice("loop_stopped"),
    ]);
    expect(turnStatus(state.turns[0])).not.toBe("streaming");
    expect(state.turns[0].terminal).toBe("cancelled");
  });

  it("does not let a voice turn close an open chat turn", () => {
    // Both surfaces run at once and neither `done` carries a run_id.
    const state = fold([
      start("gõ câu này"),
      voice("asr", "nói câu này"),
      voice("done", "trả lời nói", { terminal_status: "achieved" }),
    ]);
    const [chatTurn, voiceTurn] = state.turns;
    expect(chatTurn.finalText).toBeNull();
    expect(voiceTurn.finalText).toBe("trả lời nói");
  });

  it("does not let a chat turn close an open voice turn", () => {
    const state = fold([voice("asr", "nói"), start("gõ"), done("trả lời gõ")]);
    const [voiceTurn, chatTurn] = state.turns;
    expect(voiceTurn.finalText).toBeNull();
    expect(chatTurn.finalText).toBe("trả lời gõ");
  });

  it("ignores voice answer_delta so tokens never double the sentences", () => {
    const raw = {
      ...(delta("tok") as Record<string, unknown>),
      surface: "voice",
    } as EngineFrame;
    const state = fold([voice("asr", "hỏi"), raw, voice("sentence", "Câu.")]);
    expect(turnText(state.turns[0])).toBe("Câu.");
  });
});
