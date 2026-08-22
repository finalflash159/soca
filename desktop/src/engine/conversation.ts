/** One transcript reducer for live chat, live voice, and engine-owned snapshots. */

import type {
  EngineFrame,
  SessionSnapshotFrame,
  TerminalStatus,
  VoiceFrame,
  WorkflowFrame,
} from "./protocol";
import { isChatFrame, isVoiceFrame, isWorkflowFrame } from "./protocol";

export type TurnStatus = "streaming" | "achieved" | "blocked" | "failed";
export type TurnPageLoadState = "idle" | "loading" | "error";

/** Which surface produced a turn. Both render the same way. */
export type Surface = "chat" | "voice";

export interface Citation {
  label?: string;
  path?: string;
  [key: string]: unknown;
}

export interface Turn {
  /** Durable identity is present for restored turns and v3 live frames. */
  sessionId: string | null;
  turnId: string | null;
  sequence: number | null;
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
  /** The owner of visible history. A mismatched durable frame is dropped. */
  activeSessionId: string | null;
  persistence: "ram_only" | "local_resumable" | null;
  /** Observable protocol drift; the raw frame remains in the bounded transport log. */
  droppedSessionFrames: number;
  /** Exclusive sequence boundary for the next older durable-turn page. */
  nextTurnCursor: number | null;
  turnPageLoadState: TurnPageLoadState;
  turnPageError: string | null;
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
  activeSessionId: null,
  persistence: null,
  droppedSessionFrames: 0,
  nextTurnCursor: null,
  turnPageLoadState: "idle",
  turnPageError: null,
  reassemblyMismatch: false,
};

export type ConversationAction =
  | EngineFrame
  | { type: "turns_page_requested" }
  | { type: "turns_page_failed"; message: string };

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

function newTurn(
  runId: string,
  goalId: string,
  userText: string,
  surface: Surface,
  identity: Pick<Turn, "sessionId" | "turnId" | "sequence"> = {
    sessionId: null,
    turnId: null,
    sequence: null,
  },
): Turn {
  return {
    ...identity,
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

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function string(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function number(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function turnCursor(value: unknown): number | null {
  const cursor = number(value);
  return cursor !== null && Number.isInteger(cursor) && cursor >= 0 ? cursor : null;
}

function surface(value: unknown): Surface {
  return value === "voice" ? "voice" : "chat";
}

function restoredTurn(value: Record<string, unknown>, sessionId: string): Turn | null {
  const turnId = string(value.turn_id);
  const sequence = number(value.sequence);
  if (turnId === null || sequence === null) {
    return null;
  }
  const status = string(value.status);
  const assistantText = string(value.assistant_text);
  const errorDetail = string(value.error_detail);
  return {
    ...newTurn(
      turnId,
      "",
      string(value.user_text) ?? "",
      surface(value.surface),
      { sessionId, turnId, sequence },
    ),
    repair: string(value.repair_text),
    finalText: assistantText ?? "",
    route: string(value.route),
    blocked: value.blocked === true,
    citations: Array.isArray(value.citations) ? (value.citations as Citation[]) : [],
    terminal: string(value.terminal_status) as TerminalStatus | null,
    interrupted: status === "interrupted",
    error: status === "failed" ? errorDetail ?? "Lượt trước không hoàn tất." : null,
  };
}

/** Hydration is a data replacement, never a replay of live visual side effects. */
export function hydrateConversation(frame: SessionSnapshotFrame): ConversationState | null {
  const session = record(frame.session);
  const sessionId = session === null ? null : string(session.session_id);
  const nextTurnCursor = turnCursor(frame.next_turn_cursor);
  if (
    sessionId === null ||
    !Array.isArray(frame.turns) ||
    (frame.next_turn_cursor !== null && nextTurnCursor === null)
  ) {
    return null;
  }
  const turns: Turn[] = [];
  for (const raw of frame.turns) {
    const parsed = record(raw);
    const turn = parsed === null ? null : restoredTurn(parsed, sessionId);
    if (turn === null) return null;
    turns.push(turn);
  }
  return {
    ...initialConversation,
    activeSessionId: sessionId,
    persistence: "local_resumable",
    nextTurnCursor,
    turns,
  };
}

function frameSessionId(frame: EngineFrame): string | null {
  const direct = string((frame as Record<string, unknown>).session_id);
  if (direct !== null) {
    return direct;
  }
  const metadata = record((frame as Record<string, unknown>).metadata);
  return metadata === null ? null : string(metadata.session_id);
}

function belongsToActiveSession(state: ConversationState, frame: EngineFrame): boolean {
  const sessionId = frameSessionId(frame);
  return sessionId === null || state.activeSessionId === null || sessionId === state.activeSessionId;
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
 * Join one **chat** chunk onto the text so far.
 *
 * A chat chunk is a whole markdown block (`docs/18` §6): the engine pops at
 * blank lines, so a heading, a list, a table and a fenced snippet each arrive
 * intact. Blocks are separated by a blank line, always — that is what a block
 * *is*, and it reproduces `chat/done.text`, which keeps the newlines because it
 * is built from the raw token join.
 *
 * This used to join with a space, back when chat was fed the speech splitter's
 * sentences. Measured on a live turn, that produced:
 *
 * ```text
 * "# 4 bước fine-tune 1. Chuẩn bị dữ liệu… 2. Chọn base model…"
 * ```
 *
 * — one line, which markdown reads as a single `#` heading swallowing the whole
 * answer.
 */
function appendBlock(sofar: string, block: string): string {
  const next = block.replace(/^\s+/, "");
  if (sofar === "") {
    return next;
  }
  return `${sofar.replace(/\s+$/, "")}\n\n${next}`;
}

/**
 * Join one **voice** chunk onto the text so far.
 *
 * Voice chunks stay sentences — they are what TTS speaks, and a caption should
 * follow the speech. `pop_ready_sentence` returns `buffer[:end].strip()`, so
 * the separator between two sentences is gone and has to be put back.
 */
function appendSentence(sofar: string, chunk: string): string {
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
    const target = index >= 0 ? index : openTurnIndex(state.turns, "chat");
    if (target < 0) {
      return state;
    }
    const turn = state.turns[target];
    return patch(state, target, {
      streamedText: appendBlock(turn.streamedText, text),
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
        streamedText: appendSentence(turn.streamedText, text),
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
  action: ConversationAction,
): ConversationState {
  if ("type" in action && action.type === "turns_page_requested") {
    return state.nextTurnCursor === null
      ? state
      : { ...state, turnPageLoadState: "loading", turnPageError: null };
  }
  if ("type" in action && action.type === "turns_page_failed") {
    return {
      ...state,
      turnPageLoadState: "error",
      turnPageError:
        typeof action.message === "string" ? action.message : "Không thể tải lượt cũ hơn. Hãy thử lại.",
    };
  }

  const frame = action as EngineFrame;
  if (frame.event === "session_snapshot" || frame.event === "session_turns_page") {
    const hydrated = hydrateConversation(frame as SessionSnapshotFrame);
    if (hydrated === null) {
      return {
        ...state,
        droppedSessionFrames: state.droppedSessionFrames + 1,
        ...(frame.event === "session_turns_page"
          ? {
              turnPageLoadState: "error" as const,
              turnPageError: "Không thể đọc lượt cũ hơn; nội dung đang hiển thị không bị thay đổi.",
            }
          : {}),
      };
    }
    if (frame.event === "session_turns_page") {
      if (hydrated.activeSessionId !== state.activeSessionId) {
        return { ...state, droppedSessionFrames: state.droppedSessionFrames + 1 };
      }
      const known = new Set(state.turns.map((turn) => turn.turnId));
      return {
        ...state,
        turns: [...hydrated.turns.filter((turn) => !known.has(turn.turnId)), ...state.turns],
        nextTurnCursor: hydrated.nextTurnCursor,
        turnPageLoadState: "idle",
        turnPageError: null,
      };
    }
    return hydrated;
  }

  if (frame.event === "session_status") {
    const sessionId = string((frame as Record<string, unknown>).active_session_id);
    const persistence = (frame as Record<string, unknown>).persistence;
    return {
      ...state,
      activeSessionId: sessionId ?? state.activeSessionId,
      persistence:
        persistence === "ram_only" || persistence === "local_resumable"
          ? persistence
          : state.persistence,
    };
  }

  if (frame.event === "session_operation") {
    const operation = frame as Record<string, unknown>;
    if (
      operation.action === "create" &&
      operation.status === "completed" &&
      state.persistence === "ram_only" &&
      typeof operation.session_id === "string"
    ) {
      return {
        ...initialConversation,
        activeSessionId: operation.session_id,
        persistence: "ram_only",
      };
    }
    return state;
  }

  if (!belongsToActiveSession(state, frame)) {
    return { ...state, droppedSessionFrames: state.droppedSessionFrames + 1 };
  }

  if (isVoiceFrame(frame)) {
    return reduceVoiceTurn(state, frame);
  }

  if (isWorkflowFrame(frame)) {
    return reduceWorkflow(state, frame);
  }

  if (isChatFrame(frame)) {
    if (frame.type === "start") {
      const sessionId = frameSessionId(frame);
      return {
        ...state,
        turns: [
          ...state.turns,
          newTurn(frame.run_id ?? "", frame.goal_id ?? "", frame.text ?? "", "chat", {
            sessionId,
            turnId: typeof frame.turn_id === "string" ? frame.turn_id : null,
            sequence: typeof frame.sequence === "number" ? frame.sequence : null,
          }),
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
