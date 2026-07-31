import React from "react";
import { render } from "ink-testing-library";
import { describe, expect, it } from "vitest";
import {
  revealedGraphemes,
  splitSpeechAt,
  visibleSpeechChunks,
  VoiceStatus,
} from "./VoiceStatus.js";
import type { SpeechChunk } from "../store.js";

function chunk(
  index: number,
  text: string,
  status: SpeechChunk["status"],
  durationMs: number | null = 1000,
): SpeechChunk {
  return { index, text, durationMs, status };
}

describe("VoiceStatus speech caption", () => {
  it("splits Vietnamese text on grapheme boundaries", () => {
    expect(splitSpeechAt("chào bạn", 4)).toEqual({
      spoken: "chào",
      pending: " bạn",
    });
  });

  it("clamps the reveal boundary to the text length", () => {
    expect(splitSpeechAt("chào", -3)).toEqual({ spoken: "", pending: "chào" });
    expect(splitSpeechAt("chào", 99)).toEqual({ spoken: "chào", pending: "" });
  });

  it("never splits a word in half — holds back to the last completed word", () => {
    const text = "citation map và output";
    // Index 11 lands inside "map" ("citation m|ap"); must pull back to the
    // space before it instead of revealing "citation ma" + dim "p".
    expect(splitSpeechAt(text, 11)).toEqual({
      spoken: "citation ",
      pending: "map và output",
    });
  });

  it("does not hide an already-finished word when the cut lands on trailing punctuation", () => {
    // Qodo caught this: the first fix snapped back on ANY non-whitespace
    // character, so a cut landing on "." after a complete word wrongly
    // pulled the reveal back to the start of that word, making already
    // -spoken text disappear from the caption.
    const text = "Xin chào rồi.";
    const boundary = text.length - 1; // lands exactly on the trailing "."
    expect(splitSpeechAt(text, boundary)).toEqual({
      spoken: "Xin chào rồi",
      pending: ".",
    });
  });

  it("reveals a word in full once its own boundary is reached", () => {
    const text = "citation map và output";
    expect(splitSpeechAt(text, 12)).toEqual({
      spoken: "citation map",
      pending: " và output",
    });
  });

  it("paces the reveal by the chunk audio duration", () => {
    expect(revealedGraphemes(10, 0, 1000)).toBe(0);
    expect(revealedGraphemes(10, 500, 1000)).toBe(5);
    expect(revealedGraphemes(10, 1000, 1000)).toBe(10);
    // Playback ran longer than the synthesized audio: never overshoot.
    expect(revealedGraphemes(10, 4000, 1000)).toBe(10);
  });

  it("reveals nothing without a usable duration", () => {
    expect(revealedGraphemes(10, 500, 0)).toBe(0);
    expect(revealedGraphemes(10, 500, -1)).toBe(0);
    expect(revealedGraphemes(0, 500, 1000)).toBe(0);
  });

  it("windows the caption around the chunk being spoken", () => {
    const chunks = [
      chunk(0, "một", "complete"),
      chunk(1, "hai", "complete"),
      chunk(2, "ba", "complete"),
      chunk(3, "bốn", "playing"),
      chunk(4, "năm", "ready"),
      chunk(5, "sáu", "ready"),
    ];

    expect(visibleSpeechChunks(chunks).map((item) => item.index)).toEqual([
      2, 3, 4,
    ]);
  });

  it("windows on the last chunk once every chunk finished", () => {
    const chunks = [
      chunk(0, "một", "complete"),
      chunk(1, "hai", "complete"),
      chunk(2, "ba", "complete"),
    ];

    expect(visibleSpeechChunks(chunks).map((item) => item.index)).toEqual([
      1, 2,
    ]);
  });

  it("returns nothing when there is no speech", () => {
    expect(visibleSpeechChunks([])).toEqual([]);
  });

  it("keeps the speaking status while adding the speech caption", () => {
    const view = render(
      <VoiceStatus
        state="speaking"
        note=""
        turnIndex={2}
        latencyMs={null}
        caption={null}
        speechChunks={[chunk(0, "Mình đang trả lời bạn.", "playing", 1400)]}
        bargeIn="armed"
      />,
    );

    const frame = view.lastFrame() ?? "";
    expect(frame).toContain("speaking");
    expect(frame).toContain("barge-in armed");
    expect(frame).toContain("Mình đang trả lời bạn.");
    view.unmount();
  });

  it("renders queued chunks but drops old completed ones", () => {
    const view = render(
      <VoiceStatus
        state="speaking"
        note=""
        turnIndex={2}
        latencyMs={null}
        caption={null}
        speechChunks={[
          chunk(0, "Câu rất cũ.", "complete"),
          chunk(1, "Câu vừa nói.", "complete"),
          chunk(2, "Câu đang nói.", "playing"),
          chunk(3, "Câu sắp nói.", "ready"),
        ]}
        bargeIn="armed"
      />,
    );

    const frame = view.lastFrame() ?? "";
    expect(frame).not.toContain("Câu rất cũ.");
    expect(frame).toContain("Câu vừa nói.");
    expect(frame).toContain("Câu đang nói.");
    expect(frame).toContain("Câu sắp nói.");
    view.unmount();
  });

  it("hides the speech caption outside the speaking state", () => {
    const view = render(
      <VoiceStatus
        state="listening"
        note=""
        turnIndex={2}
        latencyMs={null}
        caption={null}
        speechChunks={[chunk(0, "Câu đang nói.", "playing")]}
        bargeIn="armed"
      />,
    );

    expect(view.lastFrame() ?? "").not.toContain("Câu đang nói.");
    view.unmount();
  });
});
