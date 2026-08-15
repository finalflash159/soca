/**
 * Voice-loop state derived from the engine's `voice` events.
 *
 * Two constraints shape this module, and both are architectural rather than
 * stylistic:
 *
 * 1. **The WebView never opens a microphone.** The engine owns capture through
 *    one duplex stream with AEC3; a second browser capture would be the
 *    two-stream arrangement that already failed on clock drift. Every level in
 *    `levels` comes from `voice_level.metadata.rms`, which the engine measures
 *    on the same buffer the recogniser sees.
 * 2. **No endpoint countdown is synthesised.** The engine publishes the floor
 *    and ceiling once, at `recording`, and never a remaining-silence figure.
 *    Counting down client-side would be inventing a decision the engine owns
 *    (`docs/18-engine-protocol.md` §7 obligation 6). Elapsed-since-last-voice is
 *    shown instead, labelled as an observation.
 */

import type { EngineFrame, VoiceFrame } from "./protocol";
import { isVoiceFrame } from "./protocol";

/** Levels retained for the meter. At ~50 frames/s this is a few seconds. */
export const LEVEL_HISTORY = 96;

export type VoicePhase =
  | "off"
  | "starting"
  | "idle"
  | "listening"
  | "transcribing"
  | "thinking"
  | "speaking";

export type BargeInState = "idle" | "armed" | "fired";

export interface EndpointConfig {
  adaptive: boolean;
  floorMs: number | null;
  ceilMs: number | null;
  smartTurn: boolean;
}

export interface PartialTranscript {
  committed: string;
  tentative: string;
}

export interface LastTurn {
  rejected: boolean;
  terminalStatus: string | null;
  durationS: number | null;
}

export interface VoiceState {
  phase: VoicePhase;
  profile: string | null;
  asrModel: string | null;
  endpoint: EndpointConfig | null;
  /** Newest last, capped at LEVEL_HISTORY. */
  levels: number[];
  partial: PartialTranscript | null;
  bargeIn: BargeInState;
  /** A rejected transcript became a Vietnamese repair prompt — not an error. */
  repairPrompt: string | null;
  lastTurn: LastTurn | null;
  turnCount: number;
  error: string | null;
}

export const initialVoice: VoiceState = {
  phase: "off",
  profile: null,
  asrModel: null,
  endpoint: null,
  levels: [],
  partial: null,
  bargeIn: "idle",
  repairPrompt: null,
  lastTurn: null,
  turnCount: 0,
  error: null,
};

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function stringOrNull(value: unknown): string | null {
  return typeof value === "string" && value !== "" ? value : null;
}

function pushLevel(levels: number[], rms: number): number[] {
  const clamped = Math.min(1, Math.max(0, rms));
  const next = levels.length >= LEVEL_HISTORY ? levels.slice(1) : levels.slice();
  next.push(clamped);
  return next;
}

function reduceVoiceFrame(state: VoiceState, frame: VoiceFrame): VoiceState {
  const metadata = frame.metadata ?? {};

  switch (frame.type) {
    case "loading":
    case "warmup":
      return { ...state, phase: "starting", error: null };

    case "ready":
      return { ...state, phase: "idle" };

    case "loop_started":
      return {
        ...state,
        phase: "idle",
        profile: stringOrNull(metadata.profile),
        error: null,
      };

    case "loop_stopped":
      // Keep the transcript and counters; only the live signals reset.
      return { ...state, phase: "off", levels: [], partial: null, bargeIn: "idle" };

    case "recording":
      return {
        ...state,
        phase: "listening",
        bargeIn: "idle",
        partial: null,
        repairPrompt: null,
        asrModel: stringOrNull(metadata.asr_model),
        endpoint: {
          adaptive: metadata.adaptive_endpoint === true,
          floorMs: numberOrNull(metadata.endpoint_floor_ms),
          ceilMs: numberOrNull(metadata.endpoint_ceil_ms),
          smartTurn: metadata.smart_turn_enabled === true,
        },
      };

    case "voice_level": {
      const rms = numberOrNull(metadata.rms);
      return rms === null ? state : { ...state, levels: pushLevel(state.levels, rms) };
    }

    case "asr_partial":
      return {
        ...state,
        partial: {
          committed: typeof metadata.committed === "string" ? metadata.committed : "",
          tentative: typeof metadata.tentative === "string" ? metadata.tentative : "",
        },
      };

    case "recorded":
      return { ...state, phase: "transcribing", levels: [] };

    case "transcribing":
      return { ...state, phase: "transcribing" };

    case "repair":
      // docs/18 §5: rejected speech becomes a repair prompt, rendered as a turn
      // rather than a failure.
      return { ...state, repairPrompt: frame.text ?? "", phase: "speaking" };

    case "turn_start":
      return { ...state, phase: "thinking" };

    case "progress":
      return state.phase === "listening" ? state : { ...state, phase: "thinking" };

    case "tts":
    case "playback_started":
      return { ...state, phase: "speaking" };

    case "barge_in": {
      const phase = stringOrNull(metadata.phase);
      if (phase === "fired") {
        return { ...state, bargeIn: "fired", phase: "listening" };
      }
      return { ...state, bargeIn: "armed" };
    }

    case "turn_end":
      return {
        ...state,
        phase: "idle",
        bargeIn: "idle",
        turnCount: state.turnCount + 1,
      };

    case "done":
      return {
        ...state,
        phase: "idle",
        lastTurn: {
          rejected: metadata.rejected === true,
          terminalStatus: stringOrNull(metadata.terminal_status),
          durationS: numberOrNull(metadata.duration_s),
        },
      };

    case "error":
      return { ...state, phase: "off", error: frame.text ?? "voice loop failed" };

    default:
      return state;
  }
}

export function reduceVoice(state: VoiceState, frame: EngineFrame): VoiceState {
  return isVoiceFrame(frame) ? reduceVoiceFrame(state, frame) : state;
}

export function voicePhaseLabel(phase: VoicePhase): string {
  switch (phase) {
    case "off":
      return "Voice off";
    case "starting":
      return "Warming up";
    case "idle":
      return "Waiting";
    case "listening":
      return "Listening";
    case "transcribing":
      return "Transcribing";
    case "thinking":
      return "Thinking";
    case "speaking":
      return "Speaking";
  }
}

/** Live transcript to show while recording; empty string when there is none. */
export function partialText(partial: PartialTranscript | null): string {
  if (partial === null) {
    return "";
  }
  return `${partial.committed} ${partial.tentative}`.trim();
}

/**
 * Peak of the retained window, for a headroom indicator.
 *
 * Deliberately not an average: a mean over a window that includes silence
 * understates speech and makes a working mic look dead.
 */
export function peakLevel(levels: number[]): number {
  let peak = 0;
  for (const level of levels) {
    if (level > peak) {
      peak = level;
    }
  }
  return peak;
}
