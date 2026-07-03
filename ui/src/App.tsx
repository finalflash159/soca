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
import { TimelineLine } from "./components/Timeline.js";
import { VoiceStatus } from "./components/VoiceStatus.js";
import { HelpOverlay } from "./components/HelpOverlay.js";
import { Bird, Wordmark } from "./components/Logo.js";
import {
  FooterHints,
  Panel,
  Spinner,
  type Hint,
} from "./components/Primitives.js";

export interface AppProps {
  mode: Mode;
  profile?: string;
  noModel?: boolean;
}

// History flows into the terminal's own scrollback via <Static> (the
// gemini-cli pattern): the brand block and finished messages render once,
// only the live controls below re-render.
type StaticItem =
  { kind: "brand" } | { kind: "entry"; entry: TimelineEntry; index: number };

function footerHints(mode: Mode, voiceRunning: boolean): Hint[] {
  const base: Hint[] = [
    { keys: "/chat /voice /status", label: "chế độ" },
    { keys: "?", label: "phím" },
    { keys: "^c", label: "thoát" },
  ];
  if (mode === "voice") {
    return [
      voiceRunning
        ? { keys: "/stop", label: "dừng nghe" }
        : { keys: "/listen", label: "bắt đầu nghe" },
      ...base,
    ];
  }
  return base;
}

function Brand({ profile, llm }: { profile: string; llm: string }) {
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
        {profile}
        {llm ? ` ${ICON.dot} ${llm}` : ""} {ICON.dot} asr · llm · tts ·
        barge-in, không cloud
      </Text>
    </Box>
  );
}

export function App({ mode: initialMode, profile, noModel = false }: AppProps) {
  const { exit } = useApp();
  const { stdout } = useStdout();
  const rawInput = Boolean(useStdin().isRawModeSupported);
  const [state, dispatch] = useReducer(reduce, {
    ...initialState,
    mode: initialMode,
  });
  const [input, setInput] = useState("");
  const [showHelp, setShowHelp] = useState(false);
  const engineRef = useRef<EngineClient | null>(null);

  const cols = stdout?.columns ?? 80;

  useEffect(() => {
    const engine = new EngineClient();
    engineRef.current = engine;
    engine.on("event", (event) => dispatch({ type: "engine_event", event }));
    engine.on("exit", () =>
      dispatch({ type: "system_message", text: "engine đã thoát" }),
    );
    engine.start({ profile, noModel });
    if (initialMode === "status") engine.send({ cmd: "status" });
    if (initialMode === "voice" && !noModel) {
      dispatch({ type: "voice_started" });
      engine.send({ cmd: "voice_start" });
    }
    return () => engine.stop();
  }, []);

  useInput(
    (char, key) => {
      if (key.escape && showHelp) setShowHelp(false);
      else if (char === "?" && input === "") setShowHelp((v) => !v);
    },
    { isActive: rawInput },
  );

  const engine = engineRef.current;

  function switchMode(next: Mode) {
    if (next !== "voice" && state.voiceRunning)
      engine?.send({ cmd: "voice_stop" });
    if (next === "status") engine?.send({ cmd: "status" });
    dispatch({ type: "set_mode", mode: next });
  }

  function onSubmit(raw: string) {
    const text = raw.trim();
    setInput("");
    if (!text) return;
    if (text.startsWith("/")) {
      const cmd = text.toLowerCase();
      if (cmd === "/quit" || cmd === "/exit") {
        engine?.stop();
        exit();
      } else if (cmd === "/chat" || cmd === "/voice" || cmd === "/status") {
        switchMode(cmd.slice(1) as Mode);
        if (cmd === "/voice" && !state.voiceRunning && !noModel) {
          dispatch({ type: "voice_started" });
          engine?.send({ cmd: "voice_start" });
        }
      } else if (cmd === "/listen") {
        dispatch({ type: "voice_started" });
        engine?.send({ cmd: "voice_start" });
      } else if (cmd === "/stop") {
        engine?.send({ cmd: "voice_stop" });
      } else if (cmd === "/memory") {
        engine?.send({ cmd: "memory" });
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

  const llm = String(state.stack["llm"] ?? "");
  const hints = useMemo(
    () => footerHints(state.mode, state.voiceRunning),
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
            <Brand key="brand" profile={state.profile} llm={llm} />
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
        />
      ) : null}

      <Box paddingX={1}>
        <Panel title={state.mode} width={cols - 2} height={2} focused>
          <Box>
            <Text color={COLOR.accent}>{`${ICON.pointer} `}</Text>
            <Box flexGrow={1}>
              <TextInput
                focus={rawInput}
                value={input}
                onChange={setInput}
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
      <Box justifyContent="space-between">
        <FooterHints hints={hints} />
        <Box paddingX={1}>
          <Text color={COLOR.muted}>
            <Text bold color={COLOR.accent}>
              {state.mode}
            </Text>
            {` ${ICON.dot} ${state.profile} ${ICON.dot} mem${noModel ? ICON.off : ICON.on}`}
          </Text>
        </Box>
      </Box>
    </Box>
  );
}

export function Splash({ onDone }: { onDone: (mode: Mode) => void }) {
  const rawInput = Boolean(useStdin().isRawModeSupported);
  useInput(
    (char, key) => {
      if (key.return) onDone("chat");
      else if (char === "v") onDone("voice");
      else if (char === "s") onDone("status");
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
          <Text color={COLOR.muted}> status</Text>
          <Text color={COLOR.muted}>{`  ${ICON.dot}  `}</Text>
          <Text color={COLOR.alt}>^c</Text>
          <Text color={COLOR.muted}> thoát</Text>
        </Text>
      </Box>
    </Box>
  );
}
