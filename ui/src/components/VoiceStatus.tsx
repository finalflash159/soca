import { useEffect, useState } from "react";
import { Box, Text, useStdout } from "ink";
import { COLOR, ICON, MUSIC_FRAMES } from "../theme.js";
import type { Caption, VoiceState } from "../store.js";

const LOADING_FRAMES = ["◐", "◓", "◑", "◒"] as const;

const STATE_VIEW: Record<VoiceState, { color: string; label: string }> = {
  loading: { color: COLOR.warn, label: "khởi động" },
  idle: { color: COLOR.muted, label: "chờ" },
  listening: { color: COLOR.accent, label: "đang nghe" },
  processing: { color: COLOR.warn, label: "đang nghĩ" },
  speaking: { color: COLOR.good, label: "đang nói" },
  error: { color: COLOR.bad, label: "lỗi" },
};

export function VoiceStatus({
  state,
  note,
  turnIndex,
  latencyMs,
  caption,
}: {
  state: VoiceState;
  note: string;
  turnIndex: number | null;
  latencyMs: number | null;
  caption: Caption | null;
}) {
  const [frame, setFrame] = useState(0);
  const animated =
    state === "speaking" || state === "loading" || state === "processing";
  useEffect(() => {
    if (!animated) return;
    const timer = setInterval(() => setFrame((f) => f + 1), 300);
    timer.unref?.();
    return () => clearInterval(timer);
  }, [animated]);

  const view = STATE_VIEW[state];
  const dot =
    state === "loading" || state === "processing"
      ? LOADING_FRAMES[frame % LOADING_FRAMES.length]
      : state === "idle"
        ? ICON.off
        : ICON.on;

  const right: string[] = [];
  if (turnIndex !== null) right.push(`lượt ${turnIndex}`);
  if (latencyMs !== null) right.push(`${(latencyMs / 1000).toFixed(1)}s`);

  const showCaption =
    caption !== null && (caption.committed !== "" || caption.tentative !== "");

  return (
    <Box flexDirection="column">
      <Box paddingX={1} justifyContent="space-between">
        <Text>
          <Text bold color={view.color}>
            {dot} {view.label}
          </Text>
          {state === "speaking" ? (
            <Text
              color={COLOR.good}
            >{`  ${MUSIC_FRAMES[frame % MUSIC_FRAMES.length]}`}</Text>
          ) : null}
          {note ? (
            <Text
              color={state === "error" ? COLOR.bad : COLOR.muted}
            >{`  ${note}`}</Text>
          ) : null}
        </Text>
        {right.length > 0 ? (
          <Text color={COLOR.muted}>{right.join(` ${ICON.dot} `)}</Text>
        ) : null}
      </Box>
      {showCaption ? (
        <Box paddingX={1}>
          <Text>
            <Text color={COLOR.text}>{caption.committed}</Text>
            {caption.tentative ? (
              <Text color={COLOR.muted}> {caption.tentative}</Text>
            ) : null}
          </Text>
        </Box>
      ) : null}
    </Box>
  );
}
