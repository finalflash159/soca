import { useEffect, useState } from "react";
import { Box, Text, useStdout } from "ink";
import { COLOR, ICON, MUSIC_FRAMES, ROLE, meterCells } from "../theme.js";
import { animationsEnabled } from "../capabilities.js";
import type { Caption, VoiceState } from "../store.js";

const LOADING_FRAMES = ["◐", "◓", "◑", "◒"] as const;

const STATE_VIEW: Record<VoiceState, { color: string; label: string }> = {
  loading: { color: ROLE.busy, label: "starting" },
  idle: { color: ROLE.idle, label: "idle" },
  listening: { color: ROLE.focus, label: "listening" },
  processing: { color: ROLE.busy, label: "thinking" },
  speaking: { color: ROLE.ok, label: "speaking" },
  error: { color: ROLE.danger, label: "error" },
};

function LevelMeter({
  value,
  width = 10,
  color,
}: {
  value: number;
  width?: number;
  color: string;
}) {
  const { filled } = meterCells(value, width);
  return (
    <Text>
      <Text color={color}>{ICON.bar.repeat(filled)}</Text>
      <Text color={COLOR.border}>
        {ICON.bar.repeat(Math.max(0, width - filled))}
      </Text>
    </Text>
  );
}

export function VoiceStatus({
  state,
  note,
  turnIndex,
  latencyMs,
  caption,
  level = 0,
  bargeIn = "off",
}: {
  state: VoiceState;
  note: string;
  turnIndex: number | null;
  latencyMs: number | null;
  caption: Caption | null;
  level?: number;
  bargeIn?: "off" | "armed" | "fired";
}) {
  const [frame, setFrame] = useState(0);
  const animated =
    animationsEnabled() &&
    (state === "speaking" || state === "loading" || state === "processing");
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
          {state === "listening" ? (
            <Text>
              {`  `}
              <LevelMeter value={level} color={ROLE.focus} />
            </Text>
          ) : null}
          {state === "speaking" ? (
            <Text
              color={COLOR.good}
            >{`  ${MUSIC_FRAMES[frame % MUSIC_FRAMES.length]}`}</Text>
          ) : null}
          {bargeIn !== "off" ? (
            <Text bold color={bargeIn === "fired" ? ROLE.ok : ROLE.busy}>
              {`  ${ICON.pointer} ${bargeIn === "fired" ? "barge-in fired" : "barge-in armed"}`}
            </Text>
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
