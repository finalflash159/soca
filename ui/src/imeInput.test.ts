import { describe, expect, it } from "vitest";
import { applyTerminalInput, graphemes, type InputKey } from "./imeInput.js";

const plain: InputKey = {};

function typeText(value: string, chunks: string[]): string {
  let state = { value, cursor: graphemes(value).length };
  for (const chunk of chunks) {
    state = applyTerminalInput(state, chunk, plain).state;
  }
  return state.value;
}

describe("terminal IME input", () => {
  it("handles replacement backspace and text arriving in one data chunk", () => {
    expect(typeText("a", ["\bá"])).toBe("á");
    expect(typeText("xabc", ["\bđ"])).toBe("xabđ");
    let replacement = applyTerminalInput(
      { value: "a", cursor: 1 },
      "",
      { delete: true },
    ).state;
    replacement = applyTerminalInput(replacement, "á", plain).state;
    expect(replacement.value).toBe("á");
  });

  it("normalizes combining input without splitting the visible character", () => {
    const result = applyTerminalInput({ value: "a", cursor: 1 }, "\u0301", plain);

    expect(result.state.value).toBe("á");
    expect(result.state.cursor).toBe(1);
    expect(applyTerminalInput(result.state, "\b", plain).state.value).toBe("");
  });

  it("moves and deletes by grapheme cluster rather than UTF-16 code unit", () => {
    const value = "Xin chào";
    const end = value.normalize("NFC").length;
    const left = applyTerminalInput({ value, cursor: end }, "", {
      rightArrow: false,
      leftArrow: true,
    });

    expect(left.state.cursor).toBe(7);
    expect(applyTerminalInput(left.state, "", { backspace: true }).state.value).toBe(
      "Xin cho",
    );
  });

  it("does not leak terminal control bytes into the prompt", () => {
    const result = applyTerminalInput(
      { value: "", cursor: 0 },
      "xin\r\nchào\t\u001b[3~",
      plain,
    );

    expect(result.state.value).toBe("xin  chào ");
  });

  it("reports Enter without changing the buffer", () => {
    const state = { value: "điều tôi phát hiện là", cursor: 20 };
    const result = applyTerminalInput(state, "", { return: true });

    expect(result.submit).toBe(true);
    expect(result.state).toEqual(state);
  });
});
