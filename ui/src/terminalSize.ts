export const DEFAULT_TERMINAL_COLUMNS = 80;
export const DEFAULT_TERMINAL_ROWS = 24;

/**
 * Terminals can briefly report an invalid size while opening or resizing.
 * Yoga expects finite, positive dimensions; never pass raw TTY values to it.
 */
export function safeTerminalDimension(
  value: number | undefined,
  fallback: number,
): number {
  if (!Number.isFinite(value) || value === undefined || value <= 0) {
    return fallback;
  }
  return Math.max(1, Math.floor(value));
}
