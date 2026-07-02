// SoCa dawn palette — must stay in sync with soca/app/style/palette.py,
// the single source of truth shared with the Python console renderers.

export const COLOR = {
  bg: "#14131c",
  surface: "#1b1a26",
  border: "#2a2838",
  text: "#e8e3d8",
  muted: "#767287",
  accent: "#e6c07b",
  accentBright: "#f2d9a4",
  accentDeep: "#b98a4e",
  alt: "#a49ac9",
  good: "#8ac79a",
  warn: "#e09652",
  bad: "#e08398",
} as const;

// clack's unicodeOr pattern: fall back to ASCII when the terminal likely
// lacks unicode glyph support (legacy Windows consoles).
const isUnicodeSupported =
  process.platform !== "win32" || Boolean(process.env["WT_SESSION"]) || process.env["TERM_PROGRAM"] === "vscode";

function u(unicode: string, ascii: string): string {
  return isUnicodeSupported ? unicode : ascii;
}

export const ICON = {
  bird: "(o>",
  birdClosed: "(.>",
  pointer: u("❯", ">"),
  dot: u("·", "-"),
  ok: u("✓", "+"),
  err: u("✗", "x"),
  bar: u("▌", "|"),
  on: u("●", "*"),
  off: u("○", "o"),
  half: u("◐", "*"),
} as const;

export const MUSIC_FRAMES = [
  "♪      ",
  "♪ ♫    ",
  "♪ ♫ ♪  ",
  "  ♫ ♪  ",
] as const;
export const SPINNER_FRAMES = [
  "⠋",
  "⠙",
  "⠹",
  "⠸",
  "⠼",
  "⠴",
  "⠦",
  "⠧",
  "⠇",
  "⠏",
] as const;

function rgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

export function lerpHex(a: string, b: string, t: number): string {
  const clamped = Math.min(1, Math.max(0, t));
  const [ar, ag, ab] = rgb(a);
  const [br, bg, bb] = rgb(b);
  const c = (x: number, y: number) =>
    Math.round(x + (y - x) * clamped)
      .toString(16)
      .padStart(2, "0");
  return `#${c(ar, br)}${c(ag, bg)}${c(ab, bb)}`;
}

export const DAWN_RAMP: readonly [string, string, string] = [
  COLOR.accentBright,
  COLOR.accent,
  COLOR.accentDeep,
];

/** Sample the dawn ramp at t in [0, 1] (first light -> gold -> horizon). */
export function rampColor(t: number): string {
  const clamped = Math.min(1, Math.max(0, t));
  return clamped <= 0.5
    ? lerpHex(DAWN_RAMP[0], DAWN_RAMP[1], clamped / 0.5)
    : lerpHex(DAWN_RAMP[1], DAWN_RAMP[2], (clamped - 0.5) / 0.5);
}
