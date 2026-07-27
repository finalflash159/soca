import { useEffect, useState } from "react";
import { useStdout } from "ink";

export interface TermSize {
  rows: number;
  cols: number;
}

export function useResize(): TermSize {
  const { stdout } = useStdout();
  const [size, setSize] = useState<TermSize>({
    rows: stdout?.rows ?? 24,
    cols: stdout?.columns ?? 80,
  });
  useEffect(() => {
    if (!stdout) return;
    let previous = { rows: stdout.rows ?? 24, cols: stdout.columns ?? 80 };
    const onResize = (): void => {
      const next = { rows: stdout.rows ?? 24, cols: stdout.columns ?? 80 };
      if (next.rows === previous.rows && next.cols === previous.cols) return;
      if (next.rows < previous.rows || next.cols < previous.cols) stdout.write("\x1b[2J\x1b[H");
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
