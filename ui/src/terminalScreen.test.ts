import { describe, expect, it } from "vitest";
import { enterAlternateScreen } from "./terminalScreen.js";

describe("terminal alternate screen", () => {
  it("enters and leaves the private fullscreen buffer exactly once", () => {
    const writes: string[] = [];
    const output = {
      isTTY: true,
      write(data: string) {
        writes.push(data);
      },
    };

    const cleanup = enterAlternateScreen(output);
    cleanup();
    cleanup();

    expect(writes).toEqual(["\u001b[?1049h\u001b[2J\u001b[H", "\u001b[?1049l"]);
  });

  it("does not emit terminal control codes for non-TTY output", () => {
    const writes: string[] = [];
    const cleanup = enterAlternateScreen({
      isTTY: false,
      write(data: string) {
        writes.push(data);
      },
    });

    cleanup();
    expect(writes).toEqual([]);
  });
});
