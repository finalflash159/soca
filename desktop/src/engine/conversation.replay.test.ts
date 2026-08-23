/**
 * Replay of a real voice turn.
 *
 * `__fixtures__/voice-turn.ndjson` was not written by hand. It is the frame
 * stream a live engine produced for one Vietnamese FLEURS clip on 2026-08-16 —
 * real Qwen ASR output, a real OpenRouter answer, real valtec TTS timings —
 * serialised through the engine's own `_sanitize`, so these lines are byte-for
 * -byte what the desktop app receives over stdout.
 *
 * Hand-written frames only ever prove the reducer agrees with my idea of the
 * protocol. This one caught three things that idea got wrong: a spoken turn was
 * dropping its route and its citations, and it read `blocked` off `rejected`
 * (an utterance the recogniser refused) rather than `runtime_blocked` (an answer
 * withheld for lack of evidence).
 */

import { describe, expect, it } from "vitest";

import { initialConversation, reduceConversation, turnStatus, turnText } from "./conversation";
import type { EngineFrame } from "./protocol";

import repairRaw from "./__fixtures__/voice-repair.ndjson?raw";
import turnRaw from "./__fixtures__/voice-turn.ndjson?raw";

function parseFrames(raw: string): EngineFrame[] {
  return raw
    .split("\n")
    .filter((line) => line.trim() !== "")
    .map((line) => JSON.parse(line) as EngineFrame);
}

const frames = parseFrames(turnRaw);
const state = frames.reduce(reduceConversation, initialConversation);

describe("a real voice turn", () => {
  it("yields exactly one spoken turn", () => {
    expect(frames).toHaveLength(7);
    expect(state.turns).toHaveLength(1);
    expect(state.turns[0].surface).toBe("voice");
  });

  it("carries the recognised utterance as the user's text", () => {
    expect(state.turns[0].userText).toContain("hiện tượng cực quan");
  });

  it("reassembles the spoken answer", () => {
    const turn = state.turns[0];
    expect(turnText(turn)).toContain("Chưa tìm thấy đủ thông tin");
    // `done.text` is authoritative on this surface (docs/18 §6), and the single
    // `sentence` frame here matches it.
    expect(turn.finalText).toBe(turnText(turn));
    expect(turn.deltaCount).toBe(1);
  });

  it("records the terminal status the engine reported", () => {
    expect(state.turns[0].terminal).toBe("insufficient_evidence");
  });

  it("keeps the route so the turn shows its provenance", () => {
    expect(state.turns[0].route).toBe("memory_llm");
  });

  it("is not blocked: the engine answered, it just had no evidence to cite", () => {
    // `runtime_blocked` is false in this fixture while `terminal_status` is
    // `insufficient_evidence`. Reading `blocked` off the terminal status would
    // hide a real spoken answer behind a refusal notice.
    expect(state.turns[0].blocked).toBe(false);
    expect(turnStatus(state.turns[0])).toBe("achieved");
  });

  it("is neither interrupted nor failed", () => {
    expect(state.turns[0].interrupted).toBe(false);
    expect(state.turns[0].error).toBeNull();
    expect(state.turns[0].repair).toBeNull();
  });

  it("ignores the frames that are not part of the transcript", () => {
    // runtime / tts / playback_started / audio all carry the answer text too.
    // Folding any of them in would repeat the answer four times over.
    const types = frames.map((frame) => (frame as { type: string }).type);
    expect(types).toEqual([
      "asr",
      "sentence",
      "runtime",
      "tts",
      "playback_started",
      "audio",
      "done",
    ]);
  });
});

/**
 * Replay of a real rejected utterance.
 *
 * `__fixtures__/voice-repair.ndjson` is a live engine's response to three
 * seconds of noise: Qwen returned an empty transcript with `no_speech`, and the
 * repair catalog answered in Vietnamese. Two behaviours here were wrong until
 * this fixture existed — the repair prompt arrived a second time as a
 * `sentence` and was printed twice, and closing the turn on `repair` threw away
 * the `needs_clarification` the engine reports on `done`.
 */
describe("a real rejected utterance", () => {
  const repairFrames = parseFrames(repairRaw);

  const repaired = repairFrames.reduce(reduceConversation, initialConversation);

  it("makes one turn with no user text", () => {
    expect(repaired.turns).toHaveLength(1);
    // The engine declined to invent a transcript, so there is nothing to show
    // as the user's message — ChatView renders no bubble for it.
    expect(repaired.turns[0].userText).toBe("");
  });

  it("carries the repair prompt exactly once", () => {
    const turn = repaired.turns[0];
    expect(turn.repair).toBe("Ơ, im ru luôn. Bạn nói lại giúp SoCa một câu nha.");
    expect(turn.streamedText).toBe("");
    expect(turnText(turn)).toBe("");
  });

  it("keeps the terminal status that says why", () => {
    expect(repaired.turns[0].terminal).toBe("needs_clarification");
  });

  it("is not an error", () => {
    // docs/18 §5: a repair is a turn, not a failure.
    expect(repaired.turns[0].error).toBeNull();
    expect(turnStatus(repaired.turns[0])).not.toBe("failed");
  });
});
