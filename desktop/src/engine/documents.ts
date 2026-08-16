/**
 * Vault documents this session has actually seen.
 *
 * The engine protocol has no "list the vault" command — `knowledge.inspect` is
 * a tool the runtime calls, not something a UI can ask for. So `@` cannot browse
 * the vault; it completes over documents that have appeared in a
 * `retrieval_trace` or a citation during this session.
 *
 * That limit is stated in the UI rather than hidden. Offering a browse
 * affordance that silently only covers seen documents would be worse than
 * offering a smaller one honestly.
 */

import type { Citation, Turn } from "./conversation";
import type { KnowledgeState } from "./knowledge";

export interface VaultDocument {
  path: string;
  /** Title from a citation, when one has been seen for this path. */
  title: string | null;
  /** Best score observed for this path, for ordering. */
  score: number;
  /** Retrieval backends that returned it. */
  backends: string[];
  /** Line range from a citation, when known. */
  lines: [number, number] | null;
}

function upsert(
  index: Map<string, VaultDocument>,
  path: string,
  patch: Partial<VaultDocument>,
): void {
  if (path === "") {
    return;
  }
  const existing = index.get(path) ?? {
    path,
    title: null,
    score: 0,
    backends: [],
    lines: null,
  };
  const backends = patch.backends
    ? Array.from(new Set([...existing.backends, ...patch.backends]))
    : existing.backends;
  index.set(path, {
    path,
    title: patch.title ?? existing.title,
    score: Math.max(existing.score, patch.score ?? 0),
    backends,
    lines: patch.lines ?? existing.lines,
  });
}

/** Build the completion index from everything seen so far. */
export function documentIndex(knowledge: KnowledgeState, turns: Turn[]): VaultDocument[] {
  const index = new Map<string, VaultDocument>();

  for (const column of knowledge.retrieval?.columns ?? []) {
    for (const hit of column.hits) {
      upsert(index, hit.path, { score: hit.score, backends: [column.source] });
    }
  }

  for (const turn of turns) {
    for (const citation of turn.citations) {
      const path = typeof citation.path === "string" ? citation.path : "";
      const title = typeof citation.title === "string" ? citation.title : null;
      const start = typeof citation.line_start === "number" ? citation.line_start : null;
      const end = typeof citation.line_end === "number" ? citation.line_end : null;
      upsert(index, path, {
        title,
        lines: start !== null && end !== null ? [start, end] : null,
      });
    }
  }

  return Array.from(index.values()).sort(
    (a, b) => b.score - a.score || a.path.localeCompare(b.path),
  );
}

/** Everything known about one cited path, for the hover preview. */
export function documentFor(documents: VaultDocument[], citation: Citation): VaultDocument | null {
  const path = typeof citation.path === "string" ? citation.path : "";
  return documents.find((document) => document.path === path) ?? null;
}

/**
 * Split the draft at an active `@` token.
 *
 * Returns null when the caret is not inside one. Only a token at a word
 * boundary counts, so an email address does not open the picker.
 */
export function mentionQuery(
  draft: string,
  caret: number,
): { start: number; query: string } | null {
  const upto = draft.slice(0, caret);
  const at = upto.lastIndexOf("@");
  if (at === -1) {
    return null;
  }
  const before = at === 0 ? " " : upto[at - 1];
  if (!/\s/.test(before)) {
    return null;
  }
  const query = upto.slice(at + 1);
  if (/\s/.test(query)) {
    return null;
  }
  return { start: at, query };
}

/** Replace the active `@` token with a path reference. */
export function applyMention(
  draft: string,
  caret: number,
  path: string,
): { text: string; caret: number } {
  const active = mentionQuery(draft, caret);
  if (active === null) {
    return { text: draft, caret };
  }
  const head = draft.slice(0, active.start);
  const tail = draft.slice(caret);
  const inserted = `@${path} `;
  return { text: `${head}${inserted}${tail}`, caret: head.length + inserted.length };
}

/** Slash commands offered in the composer palette. */
export interface SlashCommand {
  id: string;
  label: string;
  hint: string;
}

export const SLASH_COMMANDS: SlashCommand[] = [
  { id: "status", label: "/status", hint: "Runtime components and profiles" },
  { id: "context", label: "/context", hint: "Prompt budget manifest" },
  { id: "memory", label: "/memory", hint: "Working and archive memory" },
  { id: "memory_compact", label: "/compact", hint: "Compact working memory" },
  { id: "usage", label: "/usage", hint: "Session token usage" },
  { id: "knowledge_index", label: "/index", hint: "Rebuild the knowledge index" },
  { id: "llm_config", label: "/config", hint: "Active LLM configuration" },
];

/** Active `/` token, only when it starts the draft. */
export function slashQuery(draft: string): string | null {
  if (!draft.startsWith("/")) {
    return null;
  }
  const query = draft.slice(1);
  return /\s/.test(query) ? null : query;
}

export function filterCommands(query: string): SlashCommand[] {
  const needle = query.toLowerCase();
  return SLASH_COMMANDS.filter(
    (command) =>
      command.label.toLowerCase().includes(needle) || command.hint.toLowerCase().includes(needle),
  );
}
