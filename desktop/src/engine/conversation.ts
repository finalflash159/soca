/**
 * Turn assembly for the text surface.
 *
 * Implements `docs/18-engine-protocol.md` §4 (`chat`) and §6 (`answer_delta`).
 * Two rules from those sections shape everything here:
 *
 * * **Append, never replace.** Concatenating every `answer_delta` `payload.text`
 *   in order reproduces the final answer exactly. A client that replaces on each
 *   delta shows only the last chunk.
 * * **`blocked` is a terminal outcome, not an error.** Per ADR 0003 a blocked
 *   turn is the system declining to answer without evidence, so it gets its own
 *   status rather than being folded into `failed`.
 */

import type { EngineFrame, TerminalStatus, WorkflowFrame } from "./protocol";
import { isChatFrame, isWorkflowFrame } from "./protocol";

export type TurnStatus = "streaming" | "achieved" | "blocked" | "failed";

export interface Citation {
  label?: string;
  path?: string;
  [key: string]: unknown;
}

export interface Turn {
  runId: string;
  goalId: string;
  userText: string;
  /** Deltas concatenated in arrival order. Empty until the first delta lands. */
  streamedText: string;
  /** `chat/done` text — authoritative once present. */
  finalText: string | null;
  route: string | null;
  blocked: boolean;
  citations: Citation[];
  terminal: TerminalStatus | null;
  /** Latest `turn_progress.phase`; drives the UI between deltas. */
  phase: string | null;
  error: string | null;
  deltaCount: number;
}

export interface ConversationState {
  turns: Turn[];
  /**
   * True when the streamed chunks do not reassemble into the final answer,
   * ignoring the edges.
   *
   * The two cleaners differ on purpose:
   * `answer_chunk_without_citation_labels` preserves each chunk's leading and
   * trailing whitespace (stripping it would glue words together), while
   * `answer_text_without_citation_labels` ends with `.strip()`. So the
   * concatenation legitimately carries edge whitespace the final answer does
   * not, and comparing exactly fires on almost every answer.
   *
   * What is compared is the trimmed pair. Interior divergence survives that —
   * a dropped frame, or a "Nguồn:" footer the whole-answer cleaner removed and
   * the stream showed — and those are worth surfacing.
   */
  reassemblyMismatch: boolean;
}

export const initialConversation: ConversationState = {
  turns: [],
  reassemblyMismatch: false,
};

export function turnStatus(turn: Turn): TurnStatus {
  if (turn.error !== null) {
    return "failed";
  }
  if (turn.finalText === null) {
    return "streaming";
  }
  return turn.blocked ? "blocked" : "achieved";
}

/** Text to render: the streamed prefix while open, the final answer once closed. */
export function turnText(turn: Turn): string {
  return turn.finalText ?? turn.streamedText;
}

function newTurn(runId: string, goalId: string, userText: string): Turn {
  return {
    runId,
    goalId,
    userText,
    streamedText: "",
    finalText: null,
    route: null,
    blocked: false,
    citations: [],
    terminal: null,
    phase: null,
    error: null,
    deltaCount: 0,
  };
}

/** Index of the newest turn that has not reached a terminal, or -1. */
function openTurnIndex(turns: Turn[]): number {
  for (let index = turns.length - 1; index >= 0; index -= 1) {
    if (turns[index].finalText === null && turns[index].error === null) {
      return index;
    }
  }
  return -1;
}

function indexForRun(turns: Turn[], runId: string): number {
  for (let index = turns.length - 1; index >= 0; index -= 1) {
    if (turns[index].runId === runId) {
      return index;
    }
  }
  return -1;
}

/**
 * Join a chat chunk onto the text so far.
 *
 * A separator is inserted when neither side carries one, because the sentence
 * splitter strips it away. Voice chunks keep their own edges, but those never
 * reach here — see the surface guard in `reduceWorkflow`.
 */
function appendChunk(sofar: string, chunk: string): string {
  if (sofar === "") {
    return chunk;
  }
  const joined = /\s$/.test(sofar) || /^\s/.test(chunk);
  return joined ? sofar + chunk : `${sofar} ${chunk}`;
}

/** Whitespace-insensitive comparison for the reassembly check. */
function normalise(text: string): string {
  return text.replace(/\s+/g, " ").trim();
}

function patch(state: ConversationState, index: number, change: Partial<Turn>): ConversationState {
  if (index < 0) {
    return state;
  }
  const turns = state.turns.slice();
  turns[index] = { ...turns[index], ...change };
  return { ...state, turns };
}

function reduceWorkflow(state: ConversationState, frame: WorkflowFrame): ConversationState {
  if (frame.event === "answer_delta") {
    // docs/18 §6: on the voice surface a delta is a raw model token, and the
    // caption is authoritative. Only chat deltas build a chat turn.
    if (frame.surface !== "chat") {
      return state;
    }
    const text = frame.payload?.text;
    if (typeof text !== "string" || text === "") {
      return state;
    }
    // Prefer the run_id match; fall back to the open turn so a delta is never
    // dropped just because turn bookkeeping drifted.
    const index = indexForRun(state.turns, frame.run_id);
    const target = index >= 0 ? index : openTurnIndex(state.turns);
    if (target < 0) {
      return state;
    }
    const turn = state.turns[target];
    return patch(state, target, {
      streamedText: appendChunk(turn.streamedText, text),
      deltaCount: turn.deltaCount + 1,
    });
  }

  if (frame.event === "turn_terminal") {
    const index = indexForRun(state.turns, frame.run_id);
    const status = frame.payload?.status ?? frame.payload?.terminal_status;
    return patch(state, index, {
      terminal: typeof status === "string" ? (status as TerminalStatus) : null,
    });
  }

  return state;
}

export function reduceConversation(
  state: ConversationState,
  frame: EngineFrame,
): ConversationState {
  if (isWorkflowFrame(frame)) {
    return reduceWorkflow(state, frame);
  }

  if (isChatFrame(frame)) {
    if (frame.type === "start") {
      return {
        ...state,
        turns: [
          ...state.turns,
          newTurn(frame.run_id ?? "", frame.goal_id ?? "", frame.text ?? ""),
        ],
      };
    }

    if (frame.type === "done") {
      // `chat/done` carries no run_id (docs/18 §4), so it closes the open turn.
      const index = openTurnIndex(state.turns);
      if (index < 0) {
        return state;
      }
      const turn = state.turns[index];
      const finalText = frame.text ?? "";
      const mismatch =
        turn.deltaCount > 0 && normalise(turn.streamedText) !== normalise(finalText)
          ? true
          : state.reassemblyMismatch;
      const next = patch(state, index, {
        finalText,
        route: frame.route ?? null,
        blocked: frame.blocked === true,
        citations: Array.isArray(frame.citations) ? (frame.citations as Citation[]) : [],
        phase: null,
      });
      return { ...next, reassemblyMismatch: mismatch };
    }

    if (frame.type === "error") {
      const index = openTurnIndex(state.turns);
      return patch(state, index, { error: frame.text ?? "turn failed", phase: null });
    }

    return state;
  }

  if (frame.event === "turn_progress") {
    const runId = typeof frame.run_id === "string" ? frame.run_id : "";
    const index = runId !== "" ? indexForRun(state.turns, runId) : openTurnIndex(state.turns);
    const phase = typeof frame.phase === "string" ? frame.phase : null;
    return patch(state, index, { phase });
  }

  return state;
}

/** Human-readable label for the gap between phases, shown while a turn is open. */
export function phaseLabel(phase: string | null): string {
  switch (phase) {
    case "preparing":
      return "Preparing";
    case "analyzing":
      return "Analyzing";
    case "routing":
      return "Choosing capability";
    case "memory":
      return "Reading memory";
    case "retrieval":
      return "Retrieving";
    case "tool":
      return "Running tool";
    case "synthesis":
      return "Writing";
    case "validation":
      return "Verifying";
    case "speech":
      return "Speaking";
    default:
      return "Working";
  }
}

/**
 * Why a turn produced no answer.
 *
 * `blocked` alone does not say why; the terminal status does. Kept separate
 * from the error path so the UI never labels a principled refusal a failure.
 */
export function blockedReason(turn: Turn): string {
  switch (turn.terminal) {
    case "insufficient_evidence":
      return "No supporting evidence in the vault";
    case "needs_clarification":
      return "Needs a clarifying answer first";
    case "budget_exhausted":
      return "Turn budget exhausted";
    case "safe_failure":
      return "Declined for safety";
    case "cancelled":
      return "Cancelled";
    default:
      return "Declined to answer without evidence";
  }
}
