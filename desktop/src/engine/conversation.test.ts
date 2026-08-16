import { describe, expect, it } from "vitest";

import { initialConversation, reduceConversation, turnStatus, turnText } from "./conversation";
import type { EngineFrame } from "./protocol";
import { PROTOCOL_VERSION } from "./protocol";

const RUN = "run-1";

function fold(frames: EngineFrame[]) {
  return frames.reduce(reduceConversation, initialConversation);
}

const start = (text: string, runId = RUN): EngineFrame =>
  ({
    event: "chat",
    type: "start",
    text,
    run_id: runId,
    goal_id: "g",
  }) as EngineFrame;

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
  ({
    event: "chat",
    type: "done",
    text,
    route: "smalltalk",
    blocked: false,
    ...extra,
  }) as EngineFrame;

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
    // Chat chunks are markdown blocks (docs/18 §6), so each one starts a new
    // block. This assertion changed with that contract: it used to expect a
    // space, from when chat was fed the speech splitter's sentences.
    const state = fold([
      start("hỏi gì đó"),
      delta("Đoạn một."),
      delta("Đoạn hai."),
      done("Đoạn một.\n\nĐoạn hai."),
    ]);
    expect(state.turns[0].streamedText).toBe("Đoạn một.\n\nĐoạn hai.");
    expect(state.turns[0].deltaCount).toBe(2);
    expect(state.reassemblyMismatch).toBe(false);
  });

  it("does not double the separator when a chunk already carries one", () => {
    const state = fold([
      start("q"),
      delta("Nó làm khỏe cơ bắp.\n"),
      delta("  Nó tạo lực."),
      done("Nó làm khỏe cơ bắp.\n\nNó tạo lực."),
    ]);
    expect(state.turns[0].streamedText).toBe("Nó làm khỏe cơ bắp.\n\nNó tạo lực.");
    expect(state.reassemblyMismatch).toBe(false);
  });

  it("separates blocks so they never concatenate", () => {
    // Whatever the separator, it must not be nothing: raw concatenation gives
    // "Sơn Ca.Hôm nay". The reassembly check collapses whitespace, so a blank
    // line and a space both compare equal to the final answer.
    const state = fold([
      start("q"),
      delta("Xin chào! Mình là Sơn Ca."),
      delta("Hôm nay mình giúp gì được?"),
      done("Xin chào! Mình là Sơn Ca. Hôm nay mình giúp gì được?"),
    ]);
    expect(state.turns[0].streamedText).not.toContain("Ca.Hôm");
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
    // What this test is about is routing, not the separator: both deltas must
    // land on run-b and none of them on the closed turn.
    expect(state.turns[1].deltaCount).toBe(2);
    expect(state.turns[1].streamedText).toContain("second");
    expect(state.turns[1].streamedText).toContain("answer");
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
      {
        event: "turn_progress",
        run_id: RUN,
        phase: "retrieval",
      } as EngineFrame,
    ]);
    expect(state.turns[0].phase).toBe("retrieval");
  });

  it("clears the phase once the turn closes", () => {
    const state = fold([
      start("q"),
      {
        event: "turn_progress",
        run_id: RUN,
        phase: "synthesis",
      } as EngineFrame,
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

const voice = (type: string, text = "", metadata: Record<string, unknown> = {}): EngineFrame =>
  ({ event: "voice", type, text, metadata }) as EngineFrame;

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
      voice("done", "", {
        rejected: true,
        terminal_status: "needs_clarification",
      }),
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

describe("step trail", () => {
  const progress = (phase: string, runId = RUN): EngineFrame =>
    ({
      event: "turn_progress",
      phase,
      run_id: runId,
      surface: "chat",
    }) as EngineFrame;

  it("records each phase once, in order", () => {
    const state = fold([
      start("hỏi"),
      progress("preparing"),
      progress("analyzing"),
      progress("retrieval"),
      progress("synthesis"),
    ]);
    expect(state.turns[0].steps).toEqual(["preparing", "analyzing", "retrieval", "synthesis"]);
  });

  it("does not repeat a phase that reports twice", () => {
    // `turn_progress` fires per operation, not per phase change.
    const state = fold([start("hỏi"), progress("retrieval"), progress("retrieval")]);
    expect(state.turns[0].steps).toEqual(["retrieval"]);
  });

  it("leaves out `complete`, which is the closing marker not a step", () => {
    const state = fold([start("hỏi"), progress("synthesis"), progress("complete")]);
    expect(state.turns[0].steps).toEqual(["synthesis"]);
  });

  it("keeps the trail after the turn closes", () => {
    // The point of the trail is that it survives: `phase` alone is cleared on
    // `chat/done`, so a finished retrieval turn used to look like a plain one.
    const state = fold([start("hỏi"), progress("retrieval"), done("xong")]);
    expect(state.turns[0].phase).toBeNull();
    expect(state.turns[0].steps).toEqual(["retrieval"]);
  });
});

describe("markdown block boundaries in the stream", () => {
  // Measured on a live turn: `pop_ready_sentence` strips every newline, so the
  // chunks arrive as bare sentences and the client has to rebuild the blocks.
  const CHUNKS = [
    "# 4 bước fine-tune một model ngôn ngữ",
    "1. Chuẩn bị và làm sạch dữ liệu.",
    " 2. Chọn base model, tokenizer.",
    "3. Huấn luyện model trên tập train.",
  ];

  const streamed = () =>
    fold([start("hỏi"), ...CHUNKS.map((chunk) => delta(chunk))]).turns[0].streamedText;

  it("does not glue a heading onto the list that follows it", () => {
    // The bug: joined with spaces this is one line, which markdown reads as a
    // single `#` heading swallowing the entire answer.
    expect(streamed()).not.toContain("ngôn ngữ 1. Chuẩn bị");
    expect(streamed().split("\n\n")).toHaveLength(4);
  });

  it("starts each list item on its own block", () => {
    const lines = streamed().split("\n\n");
    expect(lines[1]).toBe("1. Chuẩn bị và làm sạch dữ liệu.");
    expect(lines[2]).toBe("2. Chọn base model, tokenizer.");
  });

  it("keeps a voice turn's sentences in one paragraph", () => {
    // Voice still streams sentences, not blocks — a caption follows the speech.
    const state = fold([
      voice("asr", "hỏi"),
      voice("sentence", "Câu một."),
      voice("sentence", "Câu hai."),
    ]);
    expect(state.turns[0].streamedText).toBe("Câu một. Câu hai.");
  });

  it("separates a table row, a quote and a fence too", () => {
    const state = fold([
      start("hỏi"),
      delta("Bảng:"),
      delta("| a | b |"),
      delta("> trích dẫn"),
      delta("```python"),
    ]);
    expect(state.turns[0].streamedText.split("\n\n")).toHaveLength(4);
  });

  it("does not report a reassembly mismatch for the new separator", () => {
    // The check collapses whitespace, so `\n\n` and ` ` compare equal — a
    // block-aware join must not start flagging every structured answer.
    const state = fold([
      start("hỏi"),
      delta("# Tiêu đề"),
      delta("- một"),
      done("# Tiêu đề\n\n- một"),
    ]);
    expect(state.reassemblyMismatch).toBe(false);
  });
});
