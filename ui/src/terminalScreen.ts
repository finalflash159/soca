export interface TerminalOutput {
  isTTY?: boolean;
  write(data: string): unknown;
}

const ENTER_ALTERNATE_SCREEN = "\u001b[?1049h\u001b[2J\u001b[H";
const LEAVE_ALTERNATE_SCREEN = "\u001b[?1049l";

/**
 * Put the UI in the terminal's private alternate buffer.
 *
 * The returned cleanup is idempotent so both a normal Ink shutdown and the
 * process `exit` handler can safely restore the user's previous terminal.
 */
export function enterAlternateScreen(output: TerminalOutput): () => void {
  if (!output.isTTY) return () => undefined;

  output.write(ENTER_ALTERNATE_SCREEN);
  let active = true;
  return () => {
    if (!active) return;
    active = false;
    output.write(LEAVE_ALTERNATE_SCREEN);
  };
}
