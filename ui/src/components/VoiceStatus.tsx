import { useEffect, useState } from "react";
import { Box, Text } from "ink";
import { COLOR, ICON, MUSIC_FRAMES, ROLE, meterCells } from "../theme.js";
import { animationsEnabled } from "../capabilities.js";
import { graphemes } from "../imeInput.js";
import type { Caption, SpeechChunk, VoiceState } from "../store.js";

const LOADING_FRAMES = ["◐", "◓", "◑", "◒"] as const;
const SPEECH_FRAME_MS = 80;
const CAPTION_HISTORY = 1;
const CAPTION_LOOKAHEAD = 1;

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

const WORD_CHAR_RE = /[\p{L}\p{N}]/u;

function isWordChar(grapheme: string): boolean {
  return WORD_CHAR_RE.test(grapheme);
}

/**
 * Pull a mid-word reveal boundary back to the end of the last whole word.
 *
 * revealedGraphemes() advances by a raw time ratio with no notion of word
 * boundaries, so the cut regularly lands inside a word (e.g. "map" ->
 * "ma" | "p"). Since spoken/pending render in different colors, a mid-word
 * cut is visible as a glitch rather than a typewriter effect, so the reveal
 * always holds back a partial word until it is fully spoken.
 *
 * Only snap when the cut lands ON a word character (letter/digit): that is
 * the sole case where continuing the reveal would split a token. A cut
 * landing on punctuation right after a finished word (e.g. "rồi.") must NOT
 * snap back, or the already-spoken word disappears from the caption.
 */
function snapToWordBoundary(
  parts: readonly string[],
  boundary: number,
): number {
  if (
    boundary <= 0 ||
    boundary >= parts.length ||
    !isWordChar(parts[boundary] ?? "")
  ) {
    return boundary;
  }
  let wordStart = boundary;
  while (wordStart > 0 && isWordChar(parts[wordStart - 1] ?? "")) {
    wordStart -= 1;
  }
  return wordStart;
}

/** Split spoken/pending text on grapheme boundaries, never UTF-16 code units. */
export function splitSpeechAt(
  text: string,
  revealed: number,
): { spoken: string; pending: string } {
  const parts = graphemes(text);
  const rawBoundary = Math.max(0, Math.min(parts.length, Math.floor(revealed)));
  const boundary = snapToWordBoundary(parts, rawBoundary);
  return {
    spoken: parts.slice(0, boundary).join(""),
    pending: parts.slice(boundary).join(""),
  };
}

/**
 * How much of a chunk has been spoken, paced by the chunk's own audio
 * duration. This is chunk-level sync, not word-level forced alignment.
 */
export function revealedGraphemes(
  total: number,
  elapsedMs: number,
  durationMs: number,
): number {
  if (total <= 0 || durationMs <= 0 || elapsedMs <= 0) return 0;
  return Math.min(total, Math.floor((total * elapsedMs) / durationMs));
}

/**
 * Rolling caption window: the chunk being spoken, a little already-spoken
 * context, and the queued chunk. Keeps the live region a bounded height
 * instead of growing one line per sentence for the whole turn.
 */
export function visibleSpeechChunks(
  chunks: readonly SpeechChunk[],
  { history = CAPTION_HISTORY, lookahead = CAPTION_LOOKAHEAD } = {},
): SpeechChunk[] {
  if (chunks.length === 0) return [];
  const activeAt = chunks.findIndex((chunk) => chunk.status !== "complete");
  const anchor = activeAt < 0 ? chunks.length - 1 : activeAt;
  return chunks.slice(Math.max(0, anchor - history), anchor + lookahead + 1);
}

export function VoiceStatus({
  state,
  note,
  turnIndex,
  latencyMs,
  caption,
  speechChunks,
  level = 0,
  bargeIn = "off",
}: {
  state: VoiceState;
  note: string;
  turnIndex: number | null;
  latencyMs: number | null;
  caption: Caption | null;
  speechChunks: SpeechChunk[];
  level?: number;
  bargeIn?: "off" | "armed" | "fired";
}) {
  const [frame, setFrame] = useState(0);
  const [revealed, setRevealed] = useState(0);
  const animated =
    animationsEnabled() &&
    (state === "speaking" || state === "loading" || state === "processing");
  useEffect(() => {
    if (!animated) return;
    const timer = setInterval(() => setFrame((f) => f + 1), 300);
    timer.unref?.();
    return () => clearInterval(timer);
  }, [animated]);

  const visibleChunks = visibleSpeechChunks(speechChunks);
  const playingChunk = visibleChunks.find(
    (chunk) => chunk.status === "playing",
  );
  const playingIndex = playingChunk?.index ?? null;
  const playingText = playingChunk?.text ?? "";
  const playingDurationMs = playingChunk?.durationMs ?? null;
  useEffect(() => {
    setRevealed(0);
    if (
      playingIndex === null ||
      playingDurationMs === null ||
      playingDurationMs <= 0 ||
      !animationsEnabled()
    ) {
      return;
    }
    const total = graphemes(playingText).length;
    if (total === 0) return;
    const startedAt = Date.now();
    const timer = setInterval(() => {
      const next = revealedGraphemes(
        total,
        Date.now() - startedAt,
        playingDurationMs,
      );
      setRevealed(next);
      if (next >= total) clearInterval(timer);
    }, SPEECH_FRAME_MS);
    timer.unref?.();
    return () => clearInterval(timer);
  }, [playingIndex, playingText, playingDurationMs]);

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
  const showSpeechCaption = state === "speaking" && visibleChunks.length > 0;

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
      {showSpeechCaption ? (
        <Box paddingX={1}>
          <Text>
            {visibleChunks.map((chunk, index) => {
              const prefix = index === 0 ? "" : " ";
              if (chunk.status === "complete") {
                return (
                  <Text key={chunk.index} color={COLOR.text}>
                    {prefix}
                    {chunk.text}
                  </Text>
                );
              }
              if (chunk.status === "ready") {
                return (
                  <Text key={chunk.index} color={COLOR.muted}>
                    {prefix}
                    {chunk.text}
                  </Text>
                );
              }
              const { spoken, pending } = splitSpeechAt(chunk.text, revealed);
              return (
                <Text key={chunk.index}>
                  <Text color={COLOR.text}>
                    {prefix}
                    {spoken}
                  </Text>
                  <Text color={COLOR.muted}>{pending}</Text>
                </Text>
              );
            })}
          </Text>
        </Box>
      ) : null}
    </Box>
  );
}
