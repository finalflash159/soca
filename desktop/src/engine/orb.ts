/**
 * Engine frames → static activity-orb state.
 *
 * Implements the plan's orb-state mapping, but against the vocabulary the
 * engine actually emits (`docs/18-engine-protocol.md`) rather than the prose
 * names the plan used.
 *
 * Three deviations from the plan's table, each deliberate:
 *
 * 1. **ASR transcription has no slot.** The plan's nine states cover the
 *    assistant's reasoning, not speech recognition. `transcribing` /
 *    `asr_partial` map to `working` — a component is running — because
 *    inventing a tenth state would break the §0.2 single-source rule.
 * 2. **`connecting` is time-bounded, not backend-bounded.** A remote backend is
 *    not itself an activity. `connecting` shows while the synthesis phase is
 *    open and no answer text has arrived; it becomes `composing` on the first
 *    `answer_delta`. With a local backend synthesis goes straight to
 *    `composing`.
 * 3. **Maintenance stays on its owning surface.** Knowledge indexing and
 *    memory compaction do not alter the shared Chat orb; their progress and
 *    failures are rendered on Knowledge instead. The `memory` turn phase is
 *    archive retrieval, so it remains `searching`.
 */

import type { EngineFrame, VoiceFrame, WorkflowFrame } from "./protocol";
import { isChatFrame, isVoiceFrame, isWorkflowFrame } from "./protocol";

export type OrbState =
  | "breathing"
  | "connecting"
  | "composing"
  | "listening"
  | "searching"
  | "solving"
  | "working";

/** Everything the orb needs, derived from the frame stream. */
export interface OrbActivity {
  /** Latest `turn_progress.phase`, or null between turns. */
  phase: string | null;
  /** True while a chat or voice turn is open. */
  turnOpen: boolean;
  /** True once answer text has arrived for the current turn. */
  answering: boolean;
  /** True while the mic is capturing. */
  listening: boolean;
  /** True while TTS is producing or playing audio. */
  speaking: boolean;
  /**
   * True while the voice runtime is loading, before the loop can hear anything.
   *
   * Measured on 2026-08-16: `voice_start` → first `recording` is **9.2 s** on
   * this machine, spent loading Qwen ASR, valtec TTS and warming them up. None
   * of `loading`, `warmup`, `ready` or `loop_started` reached this reducer, so
   * the orb sat on `breathing` for all nine seconds — the app looked idle and
   * ready while it was in fact busy and deaf.
   */
  voiceLoading: boolean;
  /** `remote` selects the `connecting` pre-token state. */
  backend: "local" | "remote" | null;
}

export const initialActivity: OrbActivity = {
  phase: null,
  turnOpen: false,
  answering: false,
  listening: false,
  speaking: false,
  voiceLoading: false,
  backend: null,
};

/** Phases that end a turn (`docs/18` §4 `turn_progress`). */
const CLOSING_PHASES = new Set(["complete"]);

function reduceWorkflow(activity: OrbActivity, frame: WorkflowFrame): OrbActivity {
  if (frame.event === "turn_started") {
    return { ...activity, turnOpen: true, answering: false };
  }
  if (frame.event === "answer_delta") {
    return { ...activity, answering: true };
  }
  if (frame.event === "turn_terminal") {
    return { ...activity, turnOpen: false, answering: false, phase: null };
  }
  return activity;
}

function reduceVoice(activity: OrbActivity, frame: VoiceFrame): OrbActivity {
  switch (frame.type) {
    case "loading":
    case "warmup":
      return { ...activity, voiceLoading: true };
    case "loop_started":
      // `ready` is a component signal and fires 2.2 s before the loop exists;
      // only `loop_started` means the microphone is actually open.
      return { ...activity, voiceLoading: false };
    case "recording":
    case "barge_in":
      // A barge-in hands the floor back to the user, so it is a listening state
      // even though it interrupts playback.
      return { ...activity, listening: true, speaking: false };
    case "voice_level":
      // Output-level telemetry arrives while SoCa speaks. It is not an input
      // event and must never make the shared header/chat activity claim the
      // microphone is listening.
      return frame.metadata?.source === "assistant"
        ? activity
        : { ...activity, listening: true, speaking: false };
    case "recorded":
    case "transcribing":
      return { ...activity, listening: false };
    case "tts":
    case "playback_started":
      return { ...activity, speaking: true, listening: false };
    case "turn_start":
      return { ...activity, turnOpen: true, answering: false };
    case "turn_end":
    case "done":
      return {
        ...activity,
        turnOpen: false,
        speaking: false,
        answering: false,
        phase: null,
      };
    case "error":
    case "loop_stopped":
      // A loop that failed while loading must clear the loading state, or the
      // orb spins for the rest of the session.
      return {
        ...activity,
        listening: false,
        speaking: false,
        turnOpen: false,
        voiceLoading: false,
        phase: null,
      };
    default:
      return activity;
  }
}

export function reduceActivity(activity: OrbActivity, frame: EngineFrame): OrbActivity {
  if (isWorkflowFrame(frame)) {
    return reduceWorkflow(activity, frame);
  }

  if (isVoiceFrame(frame)) {
    return reduceVoice(activity, frame);
  }

  if (isChatFrame(frame)) {
    if (frame.type === "start") {
      return { ...activity, turnOpen: true, answering: false };
    }
    if (frame.type === "done" || frame.type === "error") {
      return { ...activity, turnOpen: false, answering: false, phase: null };
    }
    return activity;
  }

  switch (frame.event) {
    case "turn_progress": {
      const phase = typeof frame.phase === "string" ? frame.phase : null;
      if (phase !== null && CLOSING_PHASES.has(phase)) {
        return { ...activity, phase: null, turnOpen: false, answering: false };
      }
      return { ...activity, phase, turnOpen: true };
    }
    case "llm_config": {
      const backend = frame.backend === "remote" ? "remote" : "local";
      return { ...activity, backend };
    }
    default:
      return activity;
  }
}

/** Phase → orb state for an open turn. */
const PHASE_STATE: Record<string, OrbState> = {
  preparing: "solving",
  analyzing: "solving",
  routing: "solving",
  memory: "searching",
  retrieval: "searching",
  tool: "working",
  validation: "solving",
  speech: "composing",
};

/**
 * Resolve one orb state.
 *
 * Order matters and is not arbitrary. This shared orb reports only the active
 * conversation. Knowledge and memory maintenance own progress UI on the
 * Knowledge surface, so they must not make Chat look busy.
 */
export function orbStateFor(activity: OrbActivity): OrbState {
  if (activity.listening) {
    return "listening";
  }
  // Above the background jobs: nine seconds of model loading with a resting orb
  // is the one state a user reads as "it is ignoring me".
  if (activity.voiceLoading) {
    return "working";
  }
  if (activity.speaking) {
    return "composing";
  }

  if (activity.phase === "synthesis") {
    if (activity.answering) {
      return "composing";
    }
    return activity.backend === "remote" ? "connecting" : "composing";
  }

  if (activity.phase !== null) {
    const mapped = PHASE_STATE[activity.phase];
    if (mapped !== undefined) {
      return mapped;
    }
  }

  if (activity.turnOpen) {
    return "solving";
  }
  return "breathing";
}

/** Short label shown next to the orb; the orb itself carries no text. */
export function orbLabel(state: OrbState): string {
  switch (state) {
    case "listening":
      return "Listening";
    case "solving":
      return "Thinking";
    case "searching":
      return "Searching";
    case "working":
      return "Processing";
    case "connecting":
      return "Connecting";
    case "composing":
      return "Responding";
    case "breathing":
      return "Idle";
  }
}
