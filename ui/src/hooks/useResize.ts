import { useEffect, useState } from "react";
import { useStdout } from "ink";
import {
  DEFAULT_TERMINAL_COLUMNS,
  DEFAULT_TERMINAL_ROWS,
  safeTerminalDimension,
} from "../terminalSize.js";

export interface TermSize {
  rows: number;
  cols: number;
}

export function useResize(): TermSize {
  const { stdout } = useStdout();
  const [size, setSize] = useState<TermSize>({
    rows: safeTerminalDimension(stdout?.rows, DEFAULT_TERMINAL_ROWS),
    cols: safeTerminalDimension(stdout?.columns, DEFAULT_TERMINAL_COLUMNS),
  });
  useEffect(() => {
    if (!stdout) return;
    let previous = {
      rows: safeTerminalDimension(stdout.rows, DEFAULT_TERMINAL_ROWS),
      cols: safeTerminalDimension(stdout.columns, DEFAULT_TERMINAL_COLUMNS),
    };
    const onResize = (): void => {
      const next = {
        rows: safeTerminalDimension(stdout.rows, DEFAULT_TERMINAL_ROWS),
        cols: safeTerminalDimension(stdout.columns, DEFAULT_TERMINAL_COLUMNS),
      };
      if (next.rows === previous.rows && next.cols === previous.cols) return;
      if (next.rows < previous.rows || next.cols < previous.cols)
        stdout.write("\x1b[2J\x1b[H");
      previous = next;
      setSize(next);
    };
    stdout.on("resize", onResize);
    return (): void => {
      stdout.off("resize", onResize);
    };
  }, [stdout]);
  return size;
}
