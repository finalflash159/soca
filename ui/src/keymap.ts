import type { Mode } from "./store.js";
import type { Hint } from "./components/Primitives.js";

export const COMMANDS = [
  "/chat",
  "/voice",
  "/status",
  "/settings",
  "/listen",
  "/stop",
  "/memory",
  "/proposals",
  "/usage",
  "/inspect",
  "/help",
  "/quit",
] as const;
export type Command = (typeof COMMANDS)[number];

export function isCommand(text: string): text is Command {
  return (COMMANDS as readonly string[]).includes(text.toLowerCase());
}

export function footerHints(mode: Mode, voiceRunning: boolean): Hint[] {
  const base: Hint[] = [
    { keys: "/chat /voice /status /settings", label: "modes" },
    { keys: "?", label: "help" },
    { keys: "^c", label: "exit" },
  ];
  return mode === "voice"
    ? [
        voiceRunning
          ? { keys: "/stop", label: "stop listening" }
          : { keys: "/listen", label: "start listening" },
        ...base,
      ]
    : base;
}
