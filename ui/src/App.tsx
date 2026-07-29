import { useEffect, useMemo, useReducer, useRef, useState } from "react";
import { Box, Static, Text, useApp, useInput, useStdin, useStdout } from "ink";
import { EngineClient } from "./engine.js";
import {
  initialState,
  reduce,
  type InfoView,
  type InteractiveMode,
  type Mode,
  type TimelineEntry,
} from "./store.js";
import { COLOR, ICON } from "./theme.js";
import {
  canonicalCommand,
  filterSlashCommands,
  footerHints as buildFooterHints,
  SLASH_COMMANDS,
} from "./keymap.js";
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
import { CommandPalette } from "./components/CommandPalette.js";
import { InformationPanel } from "./components/InformationPanel.js";
import { SessionTokenMeter } from "./components/SessionTokenMeter.js";
import { TurnProgress } from "./components/TurnProgress.js";
import { ImeTextInput } from "./imeInput.js";

export interface AppProps {
  /** The mode the user picked on the splash / CLI. */
  target: Mode;
  profile?: string;
  noModel?: boolean;
  vault?: string;
  sessionPersistence?: "ram_only" | "local_resumable";
  sessionId?: string;
  resumeSession?: boolean;
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

export function App({
  target,
  profile,
  noModel = false,
  vault,
  sessionPersistence,
  sessionId,
  resumeSession,
}: AppProps) {
  const { exit } = useApp();
  const rawInput = Boolean(useStdin().isRawModeSupported);
  // Choosing chat/voice routes through Settings first so the user picks the LLM
  // for this session; leaving Settings (Esc or picking a model) continues into
  // that mode. status/settings targets open directly.
  const gated = target === "chat" || target === "voice";
  const initialMode: InteractiveMode = gated
    ? "settings"
    : target === "status"
      ? "chat"
      : target;
  const homeMode: "chat" | "voice" = target === "voice" ? "voice" : "chat";
  const [state, dispatch] = useReducer(reduce, {
    ...initialState,
    mode: initialMode,
    activeInfo: target === "status" ? "status" : null,
  });
  const [input, setInput] = useState("");
  const [showHelp, setShowHelp] = useState(false);
  const [commandIndex, setCommandIndex] = useState(0);
  const [settingsReturnMode, setSettingsReturnMode] = useState<
    "chat" | "voice"
  >(homeMode);
  const engineRef = useRef<EngineClient | null>(null);

  const { cols } = useResize();
  // Leave one terminal column unused. Ink's log-update counts newline rows,
  // while terminals also create an implicit row when a line reaches the
  // right margin and autowraps. A full-width live border would therefore
  // leave a ghost copy behind on every keystroke.
  const safeWidth = Math.max(1, cols - 1);
  const panelWidth = Math.max(12, cols - 3);

  useEffect(() => {
    const engine = new EngineClient();
    engineRef.current = engine;
    engine.on("event", (event) => dispatch({ type: "engine_event", event }));
    engine.on("exit", () =>
      dispatch({ type: "system_message", text: "engine đã thoát" }),
    );
    engine.start({
      profile,
      noModel,
      vault,
      sessionPersistence,
      sessionId,
      resumeSession,
    });
    engine.send({ cmd: "llm_providers" });
    engine.send({ cmd: "llm_config" });
    if (target === "status") engine.send({ cmd: "status" });
    // A voice target opens in Settings first; the listening loop starts on
    // leaveSettings, not here.
    return () => engine.stop();
  }, []);

  const compactionActive =
    state.memoryCompaction?.status === "accepted" ||
    state.memoryCompaction?.status === "running";
  useEffect(() => {
    if (!compactionActive) return;
    const timer = setInterval(() => {
      engineRef.current?.send({ cmd: "memory_compact", action: "status" });
    }, 250);
    timer.unref?.();
    return () => clearInterval(timer);
  }, [compactionActive, state.memoryCompaction?.generation]);

  useEffect(() => {
    if (state.progressQueue.length === 0) return;
    const timer = setTimeout(
      () => dispatch({ type: "advance_progress" }),
      140,
    );
    timer.unref?.();
    return () => clearTimeout(timer);
  }, [state.progressQueue.length]);

  // While the help overlay is open it owns every key: the prompt is blurred
  // (see `focus` below), so any key — Esc, ?, Enter — just closes it. This keeps
  // the toggle reliable and avoids the stray-"?" bug that came from a focused
  // ImeTextInput also receives the key. Opening happens in `onPromptChange`.
  useInput(() => setShowHelp(false), { isActive: rawInput && showHelp });

  const filteredCommands = useMemo(
    () => filterSlashCommands(input),
    [input],
  );
  const commandPaletteOpen =
    input.startsWith("/") &&
    state.mode !== "settings" &&
    !showHelp &&
    !state.proposalsOpen;
  const selectedCommand =
    filteredCommands[
      Math.min(commandIndex, Math.max(0, filteredCommands.length - 1))
    ];

  useInput(
    (_character, key) => {
      if (key.escape) {
        setInput("");
        setCommandIndex(0);
        return;
      }
      if (key.upArrow) {
        setCommandIndex((value) => Math.max(0, value - 1));
        return;
      }
      if (key.downArrow) {
        setCommandIndex((value) =>
          Math.min(Math.max(0, filteredCommands.length - 1), value + 1),
        );
        return;
      }
      if (key.tab && selectedCommand) {
        setInput(
          selectedCommand.argument
            ? `${selectedCommand.value} `
            : selectedCommand.value,
        );
        setCommandIndex(0);
      }
    },
    { isActive: rawInput && commandPaletteOpen },
  );

  useInput(
    (_character, key) => {
      if (key.escape) dispatch({ type: "clear_info" });
    },
    {
      isActive:
        rawInput &&
        state.activeInfo !== null &&
        !commandPaletteOpen &&
        !showHelp &&
        !state.proposalsOpen,
    },
  );

  function onPromptChange(value: string): void {
    // Claude Code convention: "?" on an empty prompt opens the shortcuts panel
    // rather than being typed. Any other input (incl. "?" mid-message) passes
    // through untouched.
    if (!showHelp && input === "" && value === "?") {
      setShowHelp(true);
      return;
    }
    if (value && state.activeInfo !== null)
      dispatch({ type: "clear_info" });
    setCommandIndex(0);
    setInput(value);
  }

  const engine = engineRef.current;

  function switchMode(next: InteractiveMode) {
    if (
      next === "settings" &&
      (state.mode === "chat" || state.mode === "voice")
    )
      setSettingsReturnMode(state.mode);
    if (next !== "voice" && state.voiceRunning)
      engine?.send({ cmd: "voice_stop" });
    if (next === "settings") {
      engine?.send({ cmd: "llm_providers" });
      engine?.send({ cmd: "llm_config" });
    }
    dispatch({ type: "set_mode", mode: next });
  }

  // Leaving Settings returns to the interactive mode that opened it; for a
  // voice session that also restarts the listening loop.
  function leaveSettings() {
    if (settingsReturnMode === "voice" && !state.voiceRunning && !noModel) {
      dispatch({ type: "voice_started" });
      engine?.send({ cmd: "voice_start" });
    }
    switchMode(settingsReturnMode);
  }

  function showInfo(view: InfoView) {
    dispatch({ type: "show_info", view });
    if (view === "status") engine?.send({ cmd: "status" });
    else if (view === "context") {
      engine?.send({ cmd: "context" });
      engine?.send({ cmd: "usage" });
    } else if (view === "memory" || view === "compacted_summary") {
      engine?.send({ cmd: "memory" });
    } else {
      engine?.send({ cmd: "memory_compact", action: "status" });
    }
  }

  function onSubmit(raw: string) {
    let text = raw.trim();
    if (text.startsWith("/") && selectedCommand) {
      const normalized = canonicalCommand(text.toLowerCase());
      const exact =
        SLASH_COMMANDS.some((command) => command.value === normalized) ||
        normalized.startsWith("/k ");
      if (!exact || commandIndex > 0) {
        if (selectedCommand.argument) {
          setInput(`${selectedCommand.value} `);
          setCommandIndex(0);
          return;
        }
        text = selectedCommand.value;
      } else if (selectedCommand.argument && normalized === selectedCommand.value) {
        setInput(`${selectedCommand.value} `);
        return;
      }
    }
    setInput("");
    setCommandIndex(0);
    if (!text) return;
    if (text.startsWith("/")) {
      const cmd = canonicalCommand(text.toLowerCase());
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
      } else if (cmd === "/chat" || cmd === "/voice" || cmd === "/settings") {
        dispatch({ type: "clear_info" });
        switchMode(cmd.slice(1) as InteractiveMode);
        if (cmd === "/voice" && !state.voiceRunning && !noModel) {
          dispatch({ type: "voice_started" });
          engine?.send({ cmd: "voice_start" });
        }
      } else if (cmd === "/status") {
        showInfo("status");
      } else if (cmd === "/context") {
        showInfo("context");
      } else if (cmd === "/listen") {
        dispatch({ type: "clear_info" });
        switchMode("voice");
        if (!state.voiceRunning && !noModel) {
          dispatch({ type: "voice_started" });
          engine?.send({ cmd: "voice_start" });
        }
      } else if (cmd === "/stop") {
        engine?.send({ cmd: "voice_stop" });
      } else if (cmd === "/compact") {
        dispatch({ type: "show_info", view: "compaction" });
        engine?.send({ cmd: "memory_compact", action: "request" });
      } else if (cmd === "/compact-show") {
        showInfo("compacted_summary");
      } else if (cmd === "/compact-status" || cmd === "/compact-cancel") {
        dispatch({ type: "show_info", view: "compaction" });
        engine?.send({
          cmd: "memory_compact",
          action: cmd === "/compact-status" ? "status" : "cancel",
        });
      } else if (cmd === "/memory") {
        showInfo("memory");
      } else if (cmd === "/memory-proposals") {
        dispatch({ type: "clear_info" });
        engine?.send({ cmd: "memory_proposals" });
      } else if (cmd === "/help") {
        dispatch({ type: "clear_info" });
        setShowHelp(true);
      } else if (/^\/k(?:\s|$)/i.test(text)) {
        dispatch({
          type: "system_message",
          text: "/k chỉ dùng trong chat — gõ /chat rồi thử lại",
        });
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
        text: "đang ở voice mode — gõ /chat nếu muốn gửi tin nhắn văn bản",
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
    <Box flexDirection="column" width={safeWidth}>
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

      {state.activeInfo ? (
        <InformationPanel
          view={state.activeInfo}
          width={panelWidth}
          context={state.context}
          memory={state.memorySnapshot}
          usage={state.usageSnapshot}
          stack={state.stack}
          knowledge={state.knowledgeIndex}
          runtimeComponents={state.runtimeComponents}
          llmConfig={state.llmConfig}
          memoryCompaction={state.memoryCompaction}
        />
      ) : null}

      {state.turnProgress ? (
        <TurnProgress progress={state.turnProgress} />
      ) : state.chatBusy ? (
        <Box paddingX={1}>
          <Spinner label="đang gửi yêu cầu…" />
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

      {state.mode === "chat" &&
      state.timeline.length === 0 &&
      state.activeInfo === null &&
      input.length === 0 ? (
        <Empty
          icon={ICON.bird}
          title="No messages yet."
          hint="Type a question or use /voice to speak."
        />
      ) : null}

      {state.proposalsOpen ? (
        <MemoryProposalInbox
          proposals={state.proposals}
          error={state.memoryActionError}
          width={panelWidth}
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
            width={panelWidth}
            variant="idle"
          >
            <RetrievalInspector trace={state.retrievalTrace} width={panelWidth - 2} />
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
          onSelect={({
            backend,
            provider,
            model,
            max_tokens,
            reasoning_enabled,
          }) =>
            engine?.send({
              cmd: "llm_select",
              backend,
              provider,
              model,
              max_tokens,
              reasoning_enabled,
            })
          }
          onExit={leaveSettings}
        />
      ) : null}

      {state.mode !== "settings" ? (
        <>
          {commandPaletteOpen ? (
            <CommandPalette
              commands={filteredCommands}
              selectedIndex={commandIndex}
              width={panelWidth}
            />
          ) : null}
          <Box paddingX={1}>
            <Panel title={state.mode} width={panelWidth} height={2} focused>
              <Box>
                <Text color={COLOR.accent}>{`${ICON.pointer} `}</Text>
                <Box flexGrow={1}>
                  <ImeTextInput
                    focus={
                      rawInput &&
                      !showHelp &&
                      !state.proposalsOpen
                    }
                    value={input}
                    onChange={onPromptChange}
                    onSubmit={onSubmit}
                    placeholder={
                      state.mode === "voice"
                        ? "voice loop: /stop, /listen, /chat, /help…"
                        : state.chatBusy
                          ? "đang xử lý lượt hiện tại…"
                          : "nhập tin nhắn hoặc /lệnh…"
                    }
                  />
                </Box>
              </Box>
            </Panel>
          </Box>
        </>
      ) : null}
      {state.mode !== "settings" ? (
        <SessionTokenMeter stats={state.context?.session ?? null} />
      ) : null}
      <StatusBar
        hints={hints}
        mode={state.mode}
        profile={state.profile}
        memoryOn={(state.context?.session ?? null) !== null}
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
