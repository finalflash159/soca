import { useEffect, useMemo, useReducer, useRef, useState } from "react";
import { Box, Static, Text, useApp, useInput, useStdin, useStdout } from "ink";
import TextInput from "ink-text-input";
import { EngineClient } from "./engine.js";
import {
  initialState,
  reduce,
  type Mode,
  type TimelineEntry,
} from "./store.js";
import { COLOR, ICON } from "./theme.js";
import { footerHints as buildFooterHints } from "./keymap.js";
import { useResize } from "./hooks/useResize.js";
import { TimelineLine } from "./components/Timeline.js";
import { VoiceStatus } from "./components/VoiceStatus.js";
import { HelpOverlay } from "./components/HelpOverlay.js";
import { SettingsScreen } from "./components/SettingsScreen.js";
import { Bird, Wordmark } from "./components/Logo.js";
import { Empty } from "./components/Empty.js";
import { MemoryChips } from "./components/MemoryChips.js";
import { StatusBar } from "./components/StatusBar.js";
import { MemoryProposalInbox } from "./components/MemoryProposalInbox.js";
import { RetrievalInspector } from "./components/RetrievalInspector.js";
import { Panel, Spinner } from "./components/Primitives.js";

export interface AppProps {
  /** The mode the user picked on the splash / CLI. */
  target: Mode;
  profile?: string;
  noModel?: boolean;
  vault?: string;
}

// History flows into the terminal's own scrollback via <Static> (the
// gemini-cli pattern): the brand block and finished messages render once,
// only the live controls below re-render.
type StaticItem =
  { kind: "brand" } | { kind: "entry"; entry: TimelineEntry; index: number };

function Brand({ profile }: { profile: string }) {
  // No LLM model name here: this renders inside <Static> (painted once), so a
  // model shown here would freeze at startup and mislead after /settings. The
  // active backend lives in the always-live footer instead.
  return (
    <Box flexDirection="column" paddingX={1} marginBottom={1}>
      <Box marginTop={1}>
        <Bird />
      </Box>
      <Box marginTop={1}>
        <Text>
          <Wordmark />
          <Text color={COLOR.text}>
            {" "}
            — trợ lý giọng nói tiếng Việt, chạy trên máy bạn.
          </Text>
        </Text>
      </Box>
      <Text color={COLOR.muted}>
        {profile} {ICON.dot} asr · llm · tts · barge-in, không cloud
      </Text>
    </Box>
  );
}

export function App({ target, profile, noModel = false, vault }: AppProps) {
  const { exit } = useApp();
  const rawInput = Boolean(useStdin().isRawModeSupported);
  // Choosing chat/voice routes through Settings first so the user picks the LLM
  // for this session; leaving Settings (Esc or picking a model) continues into
  // that mode. status/settings targets open directly.
  const gated = target === "chat" || target === "voice";
  const initialMode: Mode = gated ? "settings" : target;
  const homeMode: Mode = gated ? target : "chat";
  const [state, dispatch] = useReducer(reduce, {
    ...initialState,
    mode: initialMode,
  });
  const [input, setInput] = useState("");
  const [showHelp, setShowHelp] = useState(false);
  const engineRef = useRef<EngineClient | null>(null);

  const { cols } = useResize();

  useEffect(() => {
    const engine = new EngineClient();
    engineRef.current = engine;
    engine.on("event", (event) => dispatch({ type: "engine_event", event }));
    engine.on("exit", () =>
      dispatch({ type: "system_message", text: "engine đã thoát" }),
    );
    engine.start({ profile, noModel, vault });
    engine.send({ cmd: "llm_providers" });
    engine.send({ cmd: "llm_config" });
    if (initialMode === "status") engine.send({ cmd: "status" });
    // A voice target opens in Settings first; the listening loop starts on
    // leaveSettings, not here.
    return () => engine.stop();
  }, []);

  // While the help overlay is open it owns every key: the prompt is blurred
  // (see `focus` below), so any key — Esc, ?, Enter — just closes it. This keeps
  // the toggle reliable and avoids the stray-"?" bug that came from a focused
  // TextInput also receiving the key. Opening happens in `onPromptChange`.
  useInput(() => setShowHelp(false), { isActive: rawInput && showHelp });

  function onPromptChange(value: string): void {
    // Claude Code convention: "?" on an empty prompt opens the shortcuts panel
    // rather than being typed. Any other input (incl. "?" mid-message) passes
    // through untouched.
    if (!showHelp && input === "" && value === "?") {
      setShowHelp(true);
      return;
    }
    setInput(value);
  }

  const engine = engineRef.current;

  function switchMode(next: Mode) {
    if (next !== "voice" && state.voiceRunning)
      engine?.send({ cmd: "voice_stop" });
    if (next === "status") engine?.send({ cmd: "status" });
    if (next === "settings") {
      engine?.send({ cmd: "llm_providers" });
      engine?.send({ cmd: "llm_config" });
    }
    dispatch({ type: "set_mode", mode: next });
  }

  // Leaving Settings continues into the session mode the user picked (homeMode);
  // for a voice session that also starts the listening loop.
  function leaveSettings() {
    if (homeMode === "voice" && !state.voiceRunning && !noModel) {
      dispatch({ type: "voice_started" });
      engine?.send({ cmd: "voice_start" });
    }
    switchMode(homeMode);
  }

  function onSubmit(raw: string) {
    const text = raw.trim();
    setInput("");
    if (!text) return;
    if (text.startsWith("/")) {
      const cmd = text.toLowerCase();
      if (state.mode === "chat" && /^\/k(?:\s|$)/i.test(text)) {
        if (text.slice(2).trim() === "") {
          dispatch({
            type: "system_message",
            text: "Cú pháp: /k <câu hỏi> — ép dùng knowledge context",
          });
        } else {
          dispatch({ type: "user_message", text });
          engine?.send({ cmd: "chat", text });
        }
        return;
      }
      if (cmd === "/quit" || cmd === "/exit") {
        engine?.stop();
        exit();
      } else if (
        cmd === "/chat" ||
        cmd === "/voice" ||
        cmd === "/status" ||
        cmd === "/settings" ||
        cmd === "/s"
      ) {
        switchMode(cmd === "/s" ? "settings" : (cmd.slice(1) as Mode));
        if (cmd === "/voice" && !state.voiceRunning && !noModel) {
          dispatch({ type: "voice_started" });
          engine?.send({ cmd: "voice_start" });
        }
      } else if (cmd === "/listen") {
        dispatch({ type: "voice_started" });
        engine?.send({ cmd: "voice_start" });
      } else if (cmd === "/stop") {
        engine?.send({ cmd: "voice_stop" });
      } else if (cmd.startsWith("/memory compact")) {
        const action = cmd.slice("/memory compact".length).trim();
        if (action === "" || action === "status" || action === "cancel") {
          engine?.send({
            cmd: "memory_compact",
            action: (action === "" ? "request" : action) as
              | "request"
              | "status"
              | "cancel",
          });
        } else {
          dispatch({
            type: "system_message",
            text: "cú pháp: /memory compact [status|cancel]",
          });
        }
      } else if (cmd === "/memory") {
        engine?.send({ cmd: "memory" });
      } else if (cmd === "/proposals") {
        engine?.send({ cmd: "memory_proposals" });
      } else if (cmd === "/usage") {
        engine?.send({ cmd: "usage" });
      } else if (cmd === "/help") {
        setShowHelp(true);
      } else {
        dispatch({
          type: "system_message",
          text: `lệnh không rõ: ${text} — gõ /help`,
        });
      }
      return;
    }
    if (state.mode !== "chat") {
      dispatch({
        type: "system_message",
        text: "đang ở chế độ chỉ xem — gõ /chat để trò chuyện",
      });
      return;
    }
    dispatch({ type: "user_message", text });
    engine?.send({ cmd: "chat", text });
  }

  const llm = state.llmConfig
    ? state.llmConfig.backend === "remote"
      ? `${state.llmConfig.provider}:${state.llmConfig.model}`
      : state.llmConfig.model
    : String(state.stack["llm"] ?? "");
  const hints = useMemo(
    () => buildFooterHints(state.mode, state.voiceRunning),
    [state.mode, state.voiceRunning],
  );
  const staticItems = useMemo<StaticItem[]>(
    () => [
      { kind: "brand" },
      ...state.timeline.map((entry, index) => ({
        kind: "entry" as const,
        entry,
        index,
      })),
    ],
    [state.timeline],
  );

  if (!state.connected) {
    return (
      <Box flexDirection="column" paddingX={1} paddingY={1}>
        <Bird />
        <Box marginTop={1}>
          <Spinner label="đang khởi động SoCa engine…" />
        </Box>
      </Box>
    );
  }

  return (
    <Box flexDirection="column" width={cols}>
      <Static items={staticItems}>
        {(item) =>
          item.kind === "brand" ? (
            <Brand key="brand" profile={state.profile} />
          ) : (
            <TimelineLine key={item.index} entry={item.entry} />
          )
        }
      </Static>

      {showHelp ? (
        <Box paddingX={1} marginBottom={1}>
          <HelpOverlay />
        </Box>
      ) : null}

      {state.mode === "status" ? (
        <Box paddingX={1} marginBottom={1} flexDirection="column">
          {state.profiles.length === 0 ? (
            <Spinner label="đang quét profile…" />
          ) : (
            state.profiles.map((p) => (
              <Box key={p.key}>
                <Box width={18} flexShrink={0}>
                  <Text bold color={COLOR.alt}>
                    {p.key}
                  </Text>
                </Box>
                <Box width={9} flexShrink={0}>
                  <Text color={p.status === "ok" ? COLOR.good : COLOR.warn}>
                    {p.status}
                  </Text>
                </Box>
                <Text color={COLOR.muted} wrap="truncate-end">
                  {p.asr} {ICON.dot} {p.llm} {ICON.dot} {p.tts}
                  {p.voice ? `/${p.voice}` : ""}
                </Text>
              </Box>
            ))
          )}
          {state.knowledgeIndex ? (
            <Text color={COLOR.muted}>
              knowledge · {state.knowledgeIndex.sparse_state} · dense {state.knowledgeIndex.dense_state} · {state.knowledgeIndex.documents} docs / {state.knowledgeIndex.chunks} chunks
            </Text>
          ) : null}
        </Box>
      ) : null}

      {state.chatBusy ? (
        <Box paddingX={1}>
          <Spinner label="SoCa đang soạn…" />
        </Box>
      ) : null}
      {state.notice ? (
        <Box paddingX={1}>
          <Text color={COLOR.muted}>
            {ICON.dot} {state.notice}
          </Text>
        </Box>
      ) : null}

      {state.mode === "voice" ? (
        <VoiceStatus
          state={state.voiceState}
          note={state.voiceNote}
          turnIndex={state.turnIndex}
          latencyMs={state.lastLatencyMs}
          caption={state.caption}
          level={state.voiceLevel}
          bargeIn={state.bargeIn}
        />
      ) : null}

      {state.mode === "chat" && state.timeline.length === 0 ? (
        <Empty
          icon={ICON.bird}
          title="No messages yet."
          hint="Type a question or use /voice to speak."
        />
      ) : null}

      {state.proposals.length > 0 ? (
        <MemoryProposalInbox
          proposals={state.proposals}
          error={state.memoryActionError}
          onApprove={(proposal_id) =>
            engine?.send({ cmd: "memory_approve", proposal_id })
          }
          onReject={(proposal_id) =>
            engine?.send({ cmd: "memory_reject", proposal_id })
          }
          onClose={() => dispatch({ type: "clear_proposals" })}
        />
      ) : null}

      {state.retrievalTrace ? (
        <Box paddingX={1} marginTop={1}>
          <Panel
            title="retrieval"
            subtitle="inspect"
            width={cols - 2}
            variant="idle"
          >
            <RetrievalInspector trace={state.retrievalTrace} width={cols - 4} />
          </Panel>
        </Box>
      ) : null}

      {state.mode !== "settings" && state.routerTier !== "none" ? (
        <Box paddingX={1} flexDirection="column">
          <Text
            color={COLOR.muted}
          >{`router ${state.routerTier} · ${state.routerLatencyMs.toFixed(1)}ms`}</Text>
          <MemoryChips
            chips={[
              {
                type: "working",
                label: "working",
                detail: `${state.memoryHits} hits`,
                active: state.memoryHits > 0,
              },
              {
                type: "semantic",
                label: "semantic",
                detail: state.memoryMode,
                active: state.memoryMode === "retrieved",
              },
              {
                type: "episodic",
                label: "episodic",
                detail: "consent gated",
                active: false,
              },
              {
                type: "procedural",
                label: "procedural",
                detail: "approval gated",
                active: false,
              },
            ]}
          />
        </Box>
      ) : null}

      {state.mode === "settings" ? (
        <SettingsScreen
          config={state.llmConfig}
          providers={state.llmProviders}
          catalog={state.llmCatalog}
          catalogProvider={state.llmCatalogProvider}
          keyPendingProvider={state.llmKeyPendingProvider}
          notice={state.settingsNotice}
          onRequestModels={(provider, query) =>
            engine?.send({ cmd: "llm_models", provider, query })
          }
          onSetKey={(provider, key) =>
            engine?.send({ cmd: "llm_set_key", provider, key })
          }
          onSelect={({ backend, provider, model }) =>
            engine?.send({ cmd: "llm_select", backend, provider, model })
          }
          onExit={leaveSettings}
        />
      ) : null}

      {state.mode !== "settings" ? (
        <Box paddingX={1}>
          <Panel title={state.mode} width={cols - 2} height={2} focused>
            <Box>
              <Text color={COLOR.accent}>{`${ICON.pointer} `}</Text>
              <Box flexGrow={1}>
                <TextInput
                  focus={rawInput && !showHelp}
                  value={input}
                  onChange={onPromptChange}
                  onSubmit={onSubmit}
                  placeholder={
                    state.mode === "voice"
                      ? "voice loop: /stop, /listen, /chat, /help…"
                      : state.chatBusy
                        ? "SoCa đang soạn câu trả lời…"
                        : "nhập tin nhắn hoặc /lệnh…"
                  }
                />
              </Box>
            </Box>
          </Panel>
        </Box>
      ) : null}
      <StatusBar
        hints={hints}
        mode={state.mode}
        profile={state.profile}
        memoryOn={!noModel}
        llm={llm}
        remote={state.llmConfig?.backend === "remote"}
      />
    </Box>
  );
}

export function Splash({ onDone }: { onDone: (mode: Mode) => void }) {
  const rawInput = Boolean(useStdin().isRawModeSupported);
  useInput(
    (char, key) => {
      if (key.return) onDone("chat");
      else if (char === "v") onDone("voice");
      else if (char === "s") onDone("settings");
    },
    { isActive: rawInput },
  );
  const { stdout } = useStdout();
  return (
    <Box
      height={Math.max(1, (stdout?.rows ?? 24) - 1)}
      flexDirection="column"
      justifyContent="center"
      alignItems="center"
    >
      <Bird />
      <Box marginTop={1}>
        <Wordmark />
      </Box>
      <Box marginTop={1}>
        <Text color={COLOR.text}>
          Trợ lý giọng nói tiếng Việt — chạy hoàn toàn trên máy bạn.
        </Text>
      </Box>
      <Box>
        <Text color={COLOR.muted}>asr · llm · tts · barge-in, không cloud</Text>
      </Box>
      <Box marginTop={1}>
        <Text>
          <Text color={COLOR.alt}>↵</Text>
          <Text color={COLOR.muted}> chat</Text>
          <Text color={COLOR.muted}>{`  ${ICON.dot}  `}</Text>
          <Text color={COLOR.alt}>v</Text>
          <Text color={COLOR.muted}> voice</Text>
          <Text color={COLOR.muted}>{`  ${ICON.dot}  `}</Text>
          <Text color={COLOR.alt}>s</Text>
          <Text color={COLOR.muted}> cài đặt</Text>
          <Text color={COLOR.muted}>{`  ${ICON.dot}  `}</Text>
          <Text color={COLOR.alt}>^c</Text>
          <Text color={COLOR.muted}> thoát</Text>
        </Text>
      </Box>
    </Box>
  );
}
