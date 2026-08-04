import { describe, expect, it } from "vitest";
import {
  DEFAULT_TERMINAL_COLUMNS,
  DEFAULT_TERMINAL_ROWS,
  safeTerminalDimension,
} from "./terminalSize.js";

describe("terminal dimensions", () => {
  it("replaces invalid TTY values with safe defaults", () => {
    expect(safeTerminalDimension(undefined, DEFAULT_TERMINAL_COLUMNS)).toBe(80);
    expect(safeTerminalDimension(Number.NaN, DEFAULT_TERMINAL_ROWS)).toBe(24);
    expect(safeTerminalDimension(0, DEFAULT_TERMINAL_ROWS)).toBe(24);
    expect(safeTerminalDimension(Number.POSITIVE_INFINITY, 24)).toBe(24);
  });

  it("normalizes valid fractional values to positive integers", () => {
    expect(safeTerminalDimension(96.9, 80)).toBe(96);
    expect(safeTerminalDimension(1, 80)).toBe(1);
  });
});
