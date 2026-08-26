/** React transport binding for the Rust sidecar manager. */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { nanoid } from "nanoid";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

import type { ConversationState } from "./conversation";
import { initialConversation, reduceConversation } from "./conversation";
import type { Citation } from "./conversation";
import type { CitationPreviewIndex } from "./citation-preview";
import {
  initialCitationPreviews,
  reduceCitationPreviews,
} from "./citation-preview";
import type { KnowledgeState } from "./knowledge";
import { initialKnowledge, reduceKnowledge } from "./knowledge";
import type { OrbActivity } from "./orb";
import { initialActivity, orbStateFor, reduceActivity } from "./orb";
import type { EngineCommand, EngineFrame, HelloFrame, StatusFrame } from "./protocol";
import { helloIsCompatible, PROTOCOL_VERSION } from "./protocol";
import type { SessionState } from "./session";
import { initialSession, reduceSession } from "./session";
import type { SessionHistoryState } from "./session-history";
import {
  initialSessionHistory,
  reduceSessionHistory,
} from "./session-history";
import type { SettingsState } from "./settings";
import { initialSettings, reduceSettings } from "./settings";
import type { VoiceState } from "./voice";
import { initialVoice, reduceVoice } from "./voice";

const EVENT_CHANNEL = "soca://engine-event";
const STATUS_CHANNEL = "soca://engine-status";
const HELLO_POLL_INTERVAL_MS = 100;
const HELLO_TIMEOUT_MS = 45_000;

/** Mirrors `LaunchOptions` in `src-tauri/src/engine.rs`. Built by `launch.ts`. */
export interface LaunchOptions {
  program?: string;
  args?: string[];
  cwd?: string;
  env?: Record<string, string>;
}

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

type MicrophonePermission = "authorized" | "denied" | "restricted";

function microphonePermissionMessage(permission: MicrophonePermission): string {
  if (permission === "denied") {
    return "SoCa cần quyền microphone. Cho phép SoCa trong System Settings > Privacy & Security > Microphone, rồi thử lại.";
  }
  return "Microphone đang bị giới hạn bởi macOS. Kiểm tra giới hạn quyền riêng tư của máy, rồi thử lại.";
}

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
  sessionHistory: SessionHistoryState;
  citationPreviews: CitationPreviewIndex;
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
  const [sessionHistory, setSessionHistory] = useState<SessionHistoryState>(initialSessionHistory);
  const [citationPreviews, setCitationPreviews] = useState<CitationPreviewIndex>(initialCitationPreviews);
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
  const helloRef = useRef<HelloFrame | null>(null);
  // `invoke` resolves after Rust has accepted the child. Keep duplicate UI
  // actions from issuing a second start while that round trip is in flight.
  const startInFlight = useRef<Promise<boolean> | null>(null);

  const acceptHello = useCallback((candidate: unknown): boolean => {
    if (
      typeof candidate !== "object" ||
      candidate === null ||
      (candidate as { event?: unknown }).event !== "hello"
    ) {
      return false;
    }
    const helloFrame = candidate as HelloFrame;
    helloRef.current = helloFrame;
    setHello(helloFrame);
    setVersionMismatch(
      helloIsCompatible(helloFrame)
        ? null
        : `engine speaks protocol ${helloFrame.protocol_version}; this app implements ${PROTOCOL_VERSION}`,
    );
    return true;
  }, []);

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
      setSessionHistory((previous) => reduceSessionHistory(previous, frame));
      setCitationPreviews((previous) => reduceCitationPreviews(previous, frame));

      if (frame.event === "hello") {
        acceptHello(frame);
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
  }, [acceptHello]);

  const start = useCallback((options?: LaunchOptions): Promise<boolean> => {
    if (startInFlight.current !== null) return startInFlight.current;

    const launch = (async () => {
      setErrors([]);
      setLog([]);
      helloRef.current = null;
      setHello(null);
      setEngineStatus(null);
      setVersionMismatch(null);
      setConversation(initialConversation);
      voiceRef.current = initialVoice;
      setVoice(initialVoice);
      setKnowledge(initialKnowledge);
      setSettings(initialSettings);
      setSession(initialSession);
      setSessionHistory(initialSessionHistory);
      setCitationPreviews(initialCitationPreviews);
      setStatus({ state: "starting", program: options?.program ?? "bundled engine" });
      try {
        await invoke("engine_start", { options: options ?? null });
        const deadline = Date.now() + HELLO_TIMEOUT_MS;
        let nativeHandshakeAvailable = true;
        while (helloRef.current === null && Date.now() < deadline) {
          const cached = await invoke<unknown>("engine_hello");
          // Undefined is a unit-test/mock response. The native command
          // serializes Option as either a frame or null.
          if (cached === undefined) {
            nativeHandshakeAvailable = false;
            break;
          }
          if (acceptHello(cached)) break;
          await new Promise<void>((resolve) => window.setTimeout(resolve, HELLO_POLL_INTERVAL_MS));
        }
        if (nativeHandshakeAvailable && helloRef.current === null) {
          await invoke("engine_stop");
          setStatus({
            state: "failed",
            message: "Engine không gửi xác nhận protocol trong 45 giây. Đã dừng tiến trình để bạn có thể thử lại.",
          });
          return false;
        }
        // The Running status also arrives as an event, but a resolved invoke is
        // proof enough that the child spawned. Do not depend on event ordering
        // for the one piece of state that gates the whole interface.
        setStatus((previous) => (previous.state === "running" ? previous : { state: "running" }));
        return true;
      } catch (error) {
        setStatus({ state: "failed", message: String(error) });
        return false;
      }
    })();

    startInFlight.current = launch;
    void launch.finally(() => {
      if (startInFlight.current === launch) startInFlight.current = null;
    });
    return launch;
  }, [acceptHello]);

  const stop = useCallback(async (): Promise<boolean> => {
    try {
      await invoke("engine_stop");
      return true;
    } catch (error) {
      setErrors((previous) => [...previous, String(error)]);
      return false;
    }
  }, []);

  const send = useCallback(async (command: EngineCommand) => {
    if (command.cmd !== "quit" && (hello === null || versionMismatch !== null)) {
      setErrors((previous) => [
        ...previous,
        versionMismatch ?? "Engine chưa xác nhận protocol tương thích.",
      ]);
      return false;
    }
    try {
      // macOS permission must originate from the signed GUI bundle. The engine
      // sidecar is intentionally not started until this one user action has an
      // explicit authorization result.
      if (command.cmd === "voice_start") {
        const permission = await invoke<MicrophonePermission>("microphone_request_access");
        if (permission !== "authorized") {
          setErrors((previous) => [...previous.slice(-19), microphonePermissionMessage(permission)]);
          return false;
        }
      }
      await invoke("engine_send", { command });
      return true;
    } catch (error) {
      setErrors((previous) => [...previous, String(error)]);
      return false;
    }
  }, [hello, versionMismatch]);

  const requestSessions = useCallback(
    async (cursor?: string) => {
      setSessionHistory((previous) =>
        reduceSessionHistory(previous, { type: "sessions_list_requested", append: cursor !== undefined }),
      );
      const sent = await send({ cmd: "sessions_list", ...(cursor === undefined ? {} : { cursor }) });
      if (!sent) {
        setSessionHistory((previous) =>
          reduceSessionHistory(previous, {
            type: "sessions_list_failed",
            message: "Không thể tải danh sách phiên đã lưu.",
          }),
        );
      }
    },
    [send],
  );

  const requestOlderTurns = useCallback(async () => {
    const cursor = conversation.nextTurnCursor;
    if (cursor === null || conversation.turnPageLoadState === "loading") return false;
    setConversation((previous) => reduceConversation(previous, { type: "turns_page_requested" }));
    const sent = await send({ cmd: "session_turns", before_sequence: cursor });
    if (!sent) {
      setConversation((previous) =>
        reduceConversation(previous, {
          type: "turns_page_failed",
          message: "Không thể tải lượt cũ hơn. Hãy thử lại.",
        }),
      );
    }
    return sent;
  }, [conversation.nextTurnCursor, conversation.turnPageLoadState, send]);

  const requestCitationPreview = useCallback(
    async (citation: Citation): Promise<boolean> => {
      const requestId = nanoid();
      setCitationPreviews((previous) =>
        reduceCitationPreviews(previous, {
          type: "citation_preview_requested",
          citation,
          requestId,
        }),
      );
      const sent = await send({
        cmd: "citation_preview",
        request_id: requestId,
        path: typeof citation.path === "string" ? citation.path : "",
        ...(typeof citation.line_start === "number" ? { line_start: citation.line_start } : {}),
        ...(typeof citation.line_end === "number" ? { line_end: citation.line_end } : {}),
        ...(typeof citation.fingerprint === "string" ? { fingerprint: citation.fingerprint } : {}),
        ...(typeof citation.source === "string" ? { source: citation.source } : {}),
      });
      if (!sent) {
        setCitationPreviews((previous) =>
          reduceCitationPreviews(previous, {
            type: "citation_preview_failed",
            citation,
            requestId,
            message: "Không thể yêu cầu kiểm tra nguồn.",
          }),
        );
      }
      return sent;
    },
    [send],
  );

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
    sessionHistory,
    citationPreviews,
    log,
    errors,
  };

  return {
    ...snapshot,
    orbState,
    start,
    stop,
    send,
    requestSessions,
    requestOlderTurns,
    requestCitationPreview,
  };
}
