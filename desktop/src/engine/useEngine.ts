/**
 * React binding for the Rust sidecar manager in `src-tauri/src/engine.rs`.
 *
 * Deliberately thin. Per `docs/18-engine-protocol.md` §7 obligation 6, routing,
 * evidence and memory-mode decisions belong to the engine — this hook keeps
 * transport state and the derived orb activity, and nothing else.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

import type { ConversationState } from "./conversation";
import { initialConversation, reduceConversation } from "./conversation";
import type { KnowledgeState } from "./knowledge";
import { initialKnowledge, reduceKnowledge } from "./knowledge";
import type { OrbActivity } from "./orb";
import { initialActivity, orbStateFor, reduceActivity } from "./orb";
import type {
  EngineCommand,
  EngineFrame,
  HelloFrame,
  StatusFrame,
} from "./protocol";
import { helloIsCompatible, PROTOCOL_VERSION } from "./protocol";
import type { SessionState } from "./session";
import { initialSession, reduceSession } from "./session";
import type { SettingsState } from "./settings";
import { initialSettings, reduceSettings } from "./settings";
import type { VoiceState } from "./voice";
import { initialVoice, reduceVoice } from "./voice";

const EVENT_CHANNEL = "soca://engine-event";
const STATUS_CHANNEL = "soca://engine-status";

/** `voice_level` arrives per audio frame; §7 obligation 5 requires throttling. */
const HIGH_FREQUENCY_EVENTS = new Set(["voice_level"]);
/** Frames kept for the on-screen log. Older ones are dropped, not paged. */
const LOG_LIMIT = 200;

export type SidecarStatus =
  | { state: "idle" }
  | { state: "starting"; program: string }
  | { state: "running" }
  | { state: "stopped"; code: number | null; graceful: boolean }
  | { state: "failed"; message: string };

export interface EngineSnapshot {
  status: SidecarStatus;
  hello: HelloFrame | null;
  /** Set when the engine speaks a protocol version this client does not implement. */
  versionMismatch: string | null;
  engineStatus: StatusFrame | null;
  activity: OrbActivity;
  conversation: ConversationState;
  voice: VoiceState;
  knowledge: KnowledgeState;
  settings: SettingsState;
  session: SessionState;
  /** Most recent frames, newest last. */
  log: EngineFrame[];
  errors: string[];
  /** True once frame and status listeners are attached. */
  ready: boolean;
}

export function useEngine() {
  const [status, setStatus] = useState<SidecarStatus>({ state: "idle" });
  const [hello, setHello] = useState<HelloFrame | null>(null);
  const [versionMismatch, setVersionMismatch] = useState<string | null>(null);
  const [engineStatus, setEngineStatus] = useState<StatusFrame | null>(null);
  const [activity, setActivity] = useState<OrbActivity>(initialActivity);
  const [conversation, setConversation] = useState<ConversationState>(initialConversation);
  const [voice, setVoice] = useState<VoiceState>(initialVoice);
  const [knowledge, setKnowledge] = useState<KnowledgeState>(initialKnowledge);
  const [settings, setSettings] = useState<SettingsState>(initialSettings);
  const [session, setSession] = useState<SessionState>(initialSession);
  const [log, setLog] = useState<EngineFrame[]>([]);
  const [errors, setErrors] = useState<string[]>([]);
  // `listen()` resolves asynchronously. Starting the engine before both
  // listeners are attached loses the first burst of frames — hello, context
  // and the Running status — and the UI then sits at `idle` forever while a
  // perfectly healthy engine talks to nobody.
  const [ready, setReady] = useState(false);

  // Activity must fold every frame, including the throttled ones, so it is kept
  // in a ref and flushed on a timer rather than driving a render per frame.
  const activityRef = useRef<OrbActivity>(initialActivity);
  const activityDirty = useRef(false);
  // `voice_level` arrives per audio frame, so voice state is folded in a ref and
  // flushed on the same timer rather than rendering once per frame (§7 obl. 5).
  const voiceRef = useRef<VoiceState>(initialVoice);
  const voiceDirty = useRef(false);

  useEffect(() => {
    const interval = window.setInterval(() => {
      if (activityDirty.current) {
        activityDirty.current = false;
        setActivity(activityRef.current);
      }
      if (voiceDirty.current) {
        voiceDirty.current = false;
        setVoice(voiceRef.current);
      }
    }, 100);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    let disposed = false;
    const unlisteners: Array<() => void> = [];

    const frameListener = listen<EngineFrame>(EVENT_CHANNEL, (event) => {
      const frame = event.payload;

      activityRef.current = reduceActivity(activityRef.current, frame);
      activityDirty.current = true;
      voiceRef.current = reduceVoice(voiceRef.current, frame);
      voiceDirty.current = true;
      setConversation((previous) => reduceConversation(previous, frame));
      setKnowledge((previous) => reduceKnowledge(previous, frame));
      setSettings((previous) => reduceSettings(previous, frame));
      setSession((previous) => reduceSession(previous, frame));

      if (frame.event === "hello") {
        const helloFrame = frame as HelloFrame;
        setHello(helloFrame);
        setVersionMismatch(
          helloIsCompatible(helloFrame)
            ? null
            : `engine speaks protocol ${helloFrame.protocol_version}; this app implements ${PROTOCOL_VERSION}`,
        );
      } else if (frame.event === "status") {
        setEngineStatus(frame as StatusFrame);
      } else if (frame.event === "engine_error") {
        const message = typeof frame.message === "string" ? frame.message : "engine error";
        const code = typeof frame.code === "string" ? ` [${frame.code}]` : "";
        setErrors((previous) => [...previous.slice(-19), `${message}${code}`]);
      }

      if (!HIGH_FREQUENCY_EVENTS.has(frame.event)) {
        setLog((previous) => [...previous.slice(-(LOG_LIMIT - 1)), frame]);
      }
    });

    const statusListener = listen<SidecarStatus>(STATUS_CHANNEL, (event) => {
      setStatus(event.payload);
      if (event.payload.state === "stopped" || event.payload.state === "failed") {
        activityRef.current = initialActivity;
        activityDirty.current = true;
        voiceRef.current = initialVoice;
        voiceDirty.current = true;
      }
    });

    void Promise.all([frameListener, statusListener]).then((attached) => {
      if (disposed) {
        for (const unlisten of attached) {
          unlisten();
        }
        return;
      }
      unlisteners.push(...attached);
      setReady(true);
    });

    return () => {
      disposed = true;
      for (const unlisten of unlisteners) {
        unlisten();
      }
    };
  }, []);

  const start = useCallback(async (options?: { program?: string; args?: string[]; cwd?: string }) => {
    setErrors([]);
    setLog([]);
    setHello(null);
    setEngineStatus(null);
    setVersionMismatch(null);
    setConversation(initialConversation);
    voiceRef.current = initialVoice;
    setVoice(initialVoice);
    setKnowledge(initialKnowledge);
    setSettings(initialSettings);
    setSession(initialSession);
    try {
      await invoke("engine_start", { options: options ?? null });
      // The Running status also arrives as an event, but a resolved invoke is
      // proof enough that the child spawned. Do not depend on event ordering
      // for the one piece of state that gates the whole interface.
      setStatus((previous) => (previous.state === "running" ? previous : { state: "running" }));
    } catch (error) {
      setStatus({ state: "failed", message: String(error) });
    }
  }, []);

  const stop = useCallback(async () => {
    try {
      await invoke("engine_stop");
    } catch (error) {
      setErrors((previous) => [...previous, String(error)]);
    }
  }, []);

  const send = useCallback(async (command: EngineCommand) => {
    try {
      await invoke("engine_send", { command });
    } catch (error) {
      setErrors((previous) => [...previous, String(error)]);
    }
  }, []);

  const orbState = useMemo(() => orbStateFor(activity), [activity]);

  const snapshot: EngineSnapshot = {
    status,
    hello,
    versionMismatch,
    engineStatus,
    activity,
    ready,
    conversation,
    voice,
    knowledge,
    settings,
    session,
    log,
    errors,
  };

  return { ...snapshot, orbState, start, stop, send };
}
