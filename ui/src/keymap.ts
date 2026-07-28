import type { Mode } from "./store.js";
import type { Hint } from "./components/Primitives.js";

export type SlashCommandGroup =
  | "mode"
  | "information"
  | "memory"
  | "voice"
  | "knowledge"
  | "application";

export interface SlashCommandDefinition {
  value: string;
  usage: string;
  description: string;
  group: SlashCommandGroup;
  argument?: boolean;
}

// One source of truth for completion, help, and dispatch validation. Aliases
// remain accepted by App, but are intentionally shown beside their canonical
// command instead of appearing as duplicate palette rows.
export const SLASH_COMMANDS: readonly SlashCommandDefinition[] = [
  {
    value: "/chat",
    usage: "/chat",
    description: "chuyển sang chat",
    group: "mode",
  },
  {
    value: "/voice",
    usage: "/voice",
    description: "chuyển sang voice",
    group: "mode",
  },
  {
    value: "/settings",
    usage: "/settings",
    description: "cài đặt LLM (alias: /s)",
    group: "mode",
  },
  {
    value: "/status",
    usage: "/status",
    description: "xem trạng thái runtime tạm thời",
    group: "information",
  },
  {
    value: "/context",
    usage: "/context",
    description: "xem context hiện tại + usage phiên (alias: /usage)",
    group: "information",
  },
  {
    value: "/memory",
    usage: "/memory",
    description: "xem working session memory",
    group: "memory",
  },
  {
    value: "/compact",
    usage: "/compact",
    description: "yêu cầu compact working memory",
    group: "memory",
  },
  {
    value: "/compact-status",
    usage: "/compact-status",
    description: "xem trạng thái compact",
    group: "memory",
  },
  {
    value: "/compact-cancel",
    usage: "/compact-cancel",
    description: "hủy compact đang chạy",
    group: "memory",
  },
  {
    value: "/compact-show",
    usage: "/compact-show",
    description: "xem summary vừa compact",
    group: "memory",
  },
  {
    value: "/memory-proposals",
    usage: "/memory-proposals",
    description: "duyệt proposal long-term memory",
    group: "memory",
  },
  {
    value: "/listen",
    usage: "/listen",
    description: "bắt đầu voice loop",
    group: "voice",
  },
  {
    value: "/stop",
    usage: "/stop",
    description: "dừng voice loop",
    group: "voice",
  },
  {
    value: "/k",
    usage: "/k <câu hỏi>",
    description: "ép truy hồi knowledge rồi đưa vào LLM",
    group: "knowledge",
    argument: true,
  },
  {
    value: "/help",
    usage: "/help",
    description: "xem toàn bộ lệnh và phím",
    group: "application",
  },
  {
    value: "/quit",
    usage: "/quit",
    description: "thoát SoCa (alias: /exit)",
    group: "application",
  },
] as const;

export const COMMAND_ALIASES: Readonly<Record<string, string>> = {
  "/s": "/settings",
  "/usage": "/context",
  "/exit": "/quit",
};

export function canonicalCommand(value: string): string {
  const normalized = value.toLowerCase();
  return COMMAND_ALIASES[normalized] ?? normalized;
}

export function filterSlashCommands(input: string): SlashCommandDefinition[] {
  const normalized = input.trimStart().toLowerCase();
  if (!normalized.startsWith("/")) return [];
  if (normalized === "/") return [...SLASH_COMMANDS];
  const aliasTarget = COMMAND_ALIASES[normalized];
  if (aliasTarget) {
    return SLASH_COMMANDS.filter((command) => command.value === aliasTarget);
  }
  const aliasTargets = new Set(
    Object.entries(COMMAND_ALIASES)
      .filter(([alias]) => alias.startsWith(normalized))
      .map(([, target]) => target),
  );
  return SLASH_COMMANDS.filter((command) => {
    if (aliasTargets.has(command.value)) return true;
    if (command.value.startsWith(normalized)) return true;
    if (command.argument && normalized.startsWith(`${command.value} `))
      return true;
    return false;
  });
}

export function footerHints(mode: Mode, voiceRunning: boolean): Hint[] {
  const base: Hint[] = [
    { keys: "/", label: "commands" },
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
