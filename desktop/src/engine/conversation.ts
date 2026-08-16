/**
 * Turn assembly — one transcript, both surfaces.
 *
 * Implements `docs/18-engine-protocol.md` §4 (`chat`), §5 (`voice`) and §6
 * (`answer_delta`). Three rules from those sections shape everything here:
 *
 * * **Append, never replace.** Concatenating every `answer_delta` `payload.text`
 *   in order reproduces the final answer exactly. A client that replaces on each
 *   delta shows only the last chunk.
 * * **`blocked` is a terminal outcome, not an error.** Per ADR 0003 a blocked
 *   turn is the system declining to answer without evidence, so it gets its own
 *   status rather than being folded into `failed`.
 * * **A spoken turn is a turn.** Voice and chat produce the same `Turn`, tagged
 *   by `surface`, in one ordered list.
 *
 * That last rule is why this file grew. Voice used to reduce into a separate
 * live-signals-only state, so the app had no spoken history at all: the moment
 * a voice turn ended, everything said in it was gone. The transcript was never
 * missing from the wire — `voice/asr` carries the recognised user text and
 * `voice/sentence` carries each guardrail-passed answer sentence — the client
 * simply dropped both.
 *
 * Voice text comes from `voice/sentence`, not from `answer_delta`. On the voice
 * surface a delta is a raw model token (§6), so it can carry a citation label
 * the final text strips and it breaks mid-word; `sentence` is the same text TTS
 * speaks, which is what a caption should show.
 */

import type { EngineFrame, TerminalStatus, VoiceFrame, WorkflowFrame } from "./protocol";
import { isChatFrame, isVoiceFrame, isWorkflowFrame } from "./protocol";

export type TurnStatus = "streaming" | "achieved" | "blocked" | "failed";

/** Which surface produced a turn. Both render the same way. */
export type Surface = "chat" | "voice";

export interface Citation {
  label?: string;
  path?: string;
  [key: string]: unknown;
}

export interface Turn {
  runId: string;
  goalId: string;
  surface: Surface;
  userText: string;
  /**
   * A repair prompt shown in place of an answer (§5).
   *
   * Rejected speech becomes a Vietnamese question rather than an invented
   * transcript, so this is a turn outcome, never an error.
   */
  repair: string | null;
  /** Barge-in cut the answer short; what was said stands, but it is incomplete. */
  interrupted: boolean;
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
  /**
   * Every distinct phase this turn passed through, in order.
   *
   * `phase` alone is a single latest value, so once a turn finished there was
   * no record of what it had actually done — a retrieval turn and a plain chat
   * turn were indistinguishable after the fact. This is the trail plan §5.6.4
   * asks for, built from frames the engine already sends.
   */
  steps: string[];
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

function newTurn(runId: string, goalId: string, userText: string, surface: Surface): Turn {
  return {
    runId,
    goalId,
    surface,
    userText,
    repair: null,
    interrupted: false,
    streamedText: "",
    finalText: null,
    route: null,
    blocked: false,
    citations: [],
    terminal: null,
    phase: null,
    steps: [],
    error: null,
    deltaCount: 0,
  };
}

/**
 * Index of the newest open turn on a surface, or -1.
 *
 * Scoped by surface because the two interleave: talking while a chat turn is
 * still streaming must not let `voice/done` close the typed turn, and the
 * reverse. Neither `chat/done` nor `voice/done` carries a `run_id` to match on.
 */
function openTurnIndex(turns: Turn[], surface: Surface): number {
  for (let index = turns.length - 1; index >= 0; index -= 1) {
    const turn = turns[index];
    if (turn.surface === surface && turn.finalText === null && turn.error === null) {
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
 * A chunk that opens a new markdown block rather than continuing a sentence.
 *
 * Headings, list items of both kinds, block quotes, fences, tables and rules.
 */
const BLOCK_START = /^\s*(#{1,6}\s|[-*+]\s|\d+[.)]\s|>\s|```|~~~|\||---\s*$)/;

/**
 * Join a chat chunk onto the text so far.
 *
 * `pop_ready_sentence` returns `buffer[:end].strip()`, so **every newline the
 * model wrote is gone by the time a chunk reaches a client** — the separator
 * has to be reconstructed here. Joining with a space was right for prose and
 * catastrophic for anything else. Measured on a live turn:
 *
 * ```text
 * chunks : ["# 4 bước fine-tune", "1. Chuẩn bị dữ liệu…", " 2. Chọn base model…"]
 * space  → "# 4 bước fine-tune 1. Chuẩn bị dữ liệu… 2. Chọn base model…"
 * ```
 *
 * That is one line, and markdown reads it as a single `#` heading swallowing
 * the whole answer. The blank line is what makes it three blocks again, and it
 * reproduces `chat/done.text`, which keeps the newlines because it is built
 * from the raw token join.
 *
 * Prose is unchanged: a chunk that does not open a block still joins with a
 * space, so consecutive sentences stay in one paragraph.
 *
 * Voice chunks keep their own edges and never reach here — see the surface
 * guard in `reduceWorkflow`.
 */
function appendChunk(sofar: string, chunk: string): string {
  if (sofar === "") {
    return chunk;
  }
  if (BLOCK_START.test(chunk)) {
    return `${sofar.replace(/\s+$/, "")}\n\n${chunk.replace(/^\s+/, "")}`;
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
    const target = index >= 0 ? index : openTurnIndex(state.turns, "chat");
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

/**
 * Build spoken turns from the voice event stream.
 *
 * The shape of a voice turn on the wire (§5), in order:
 *
 * ```text
 * asr        text = recognised utterance ("" when nothing was understood)
 * sentence   text = one guardrail-passed answer sentence, repeated
 * done       text = the authoritative full answer
 * ```
 *
 * `repair` replaces the answer when the utterance was rejected, and
 * `interrupted` marks an answer that barge-in cut short.
 */
function reduceVoiceTurn(state: ConversationState, frame: VoiceFrame): ConversationState {
  const text = typeof frame.text === "string" ? frame.text : "";
  const metadata = frame.metadata ?? {};

  switch (frame.type) {
    case "asr": {
      // An empty transcript is not a turn yet: `repair` decides whether the
      // engine asks again, and inventing a blank user bubble here would show
      // one for every cough the VAD opened on.
      if (text.trim() === "") {
        return state;
      }
      return {
        ...state,
        turns: [...state.turns, newTurn("", "", text, "voice")],
      };
    }

    case "repair": {
      // The turn stays open. Closing it here would be simpler, but a measured
      // repair turn ends with `done.terminal_status = needs_clarification`, and
      // a closed turn cannot record it.
      const index = openTurnIndex(state.turns, "voice");
      if (index >= 0) {
        return patch(state, index, { repair: text });
      }
      // Rejected before any transcript existed — the common case. The turn is
      // the engine asking again, with nothing recognised on the user's side.
      const turn = { ...newTurn("", "", "", "voice"), repair: text };
      return { ...state, turns: [...state.turns, turn] };
    }

    case "sentence": {
      const index = openTurnIndex(state.turns, "voice");
      if (index < 0 || text === "") {
        return state;
      }
      const turn = state.turns[index];
      // A repair is spoken through the same TTS path, so the prompt arrives a
      // second time as a `sentence`. Measured, not guessed — see the replay
      // fixture. Folding it in would print the question twice.
      if (turn.repair !== null) {
        return state;
      }
      return patch(state, index, {
        streamedText: appendChunk(turn.streamedText, text),
        deltaCount: turn.deltaCount + 1,
      });
    }

    case "interrupted": {
      const index = openTurnIndex(state.turns, "voice");
      return patch(state, index, { interrupted: true });
    }

    case "done": {
      const index = openTurnIndex(state.turns, "voice");
      if (index < 0) {
        return state;
      }
      const status = metadata.terminal_status;
      const route = metadata.runtime_route;
      return patch(state, index, {
        // A rejected turn already showed its repair prompt; keeping the empty
        // `done.text` would blank it out.
        finalText: state.turns[index].repair !== null ? "" : text,
        terminal: typeof status === "string" ? (status as TerminalStatus) : null,
        // `runtime_blocked`, not `rejected`. Measured against a real turn: a
        // `rejected` utterance is one the recogniser refused, which the repair
        // prompt already covers, whereas withholding an answer for lack of
        // evidence is `runtime_blocked` — the voice twin of `chat/done.blocked`.
        blocked: metadata.runtime_blocked === true,
        // Route and citations reach the chat surface through `chat/done` and
        // reached voice through nothing at all, so a spoken turn used to render
        // with no provenance line even when it had cited something.
        route: typeof route === "string" ? route : null,
        citations: Array.isArray(metadata.citations) ? (metadata.citations as Citation[]) : [],
        interrupted: state.turns[index].interrupted || metadata.interrupted === true,
        phase: null,
      });
    }

    case "error": {
      const index = openTurnIndex(state.turns, "voice");
      return patch(state, index, {
        error: text !== "" ? text : "voice turn failed",
      });
    }

    case "loop_stopped": {
      // Stopping mid-turn leaves a bubble that would spin forever.
      const index = openTurnIndex(state.turns, "voice");
      if (index < 0) {
        return state;
      }
      const turn = state.turns[index];
      return patch(state, index, {
        finalText: turn.streamedText,
        interrupted: turn.streamedText !== "" || turn.repair !== null,
        terminal: turn.terminal ?? "cancelled",
        phase: null,
      });
    }

    default:
      return state;
  }
}

export function reduceConversation(
  state: ConversationState,
  frame: EngineFrame,
): ConversationState {
  if (isVoiceFrame(frame)) {
    return reduceVoiceTurn(state, frame);
  }

  if (isWorkflowFrame(frame)) {
    return reduceWorkflow(state, frame);
  }

  if (isChatFrame(frame)) {
    if (frame.type === "start") {
      return {
        ...state,
        turns: [
          ...state.turns,
          newTurn(frame.run_id ?? "", frame.goal_id ?? "", frame.text ?? "", "chat"),
        ],
      };
    }

    if (frame.type === "done") {
      // `chat/done` carries no run_id (docs/18 §4), so it closes the open
      // turn — the open *chat* turn, since a voice turn may be open too.
      const index = openTurnIndex(state.turns, "chat");
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
      const index = openTurnIndex(state.turns, "chat");
      return patch(state, index, {
        error: frame.text ?? "turn failed",
        phase: null,
      });
    }

    return state;
  }

  if (frame.event === "turn_progress") {
    const runId = typeof frame.run_id === "string" ? frame.run_id : "";
    const surface = frame.surface === "voice" ? "voice" : "chat";
    const index =
      runId !== "" ? indexForRun(state.turns, runId) : openTurnIndex(state.turns, surface);
    const phase = typeof frame.phase === "string" ? frame.phase : null;
    if (index < 0) {
      return state;
    }
    // `complete` is the closing marker, not work (see CLOSING_PHASES in orb.ts);
    // recording it would put a step in the trail that never happened.
    const steps = state.turns[index].steps;
    const record = phase !== null && phase !== "complete" && steps[steps.length - 1] !== phase;
    return patch(state, index, {
      phase,
      steps: record ? [...steps, phase] : steps,
    });
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
