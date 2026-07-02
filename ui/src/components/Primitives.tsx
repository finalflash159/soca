import { useEffect, useState, type ReactNode } from "react";
import { Box, Text, useStdout } from "ink";
import { COLOR, SPINNER_FRAMES } from "../theme.js";

export function Rule({ width }: { width: number }) {
  return <Text color={COLOR.border}>{"─".repeat(Math.max(1, width))}</Text>;
}

export function Spinner({ label }: { label?: string }) {
  const [frame, setFrame] = useState(0);
  useEffect(() => {
    const timer = setInterval(
      () => setFrame((f) => (f + 1) % SPINNER_FRAMES.length),
      80,
    );
    timer.unref?.();
    return () => clearInterval(timer);
  }, []);
  return (
    <Text>
      <Text color={COLOR.accent}>{SPINNER_FRAMES[frame]}</Text>
      {label ? <Text color={COLOR.muted}>{` ${label}`}</Text> : null}
    </Text>
  );
}

export interface Hint {
  keys: string;
  label: string;
}

export function FooterHints({ hints }: { hints: Hint[] }) {
  return (
    <Box paddingX={1}>
      <Text>
        {hints.map((h, i) => (
          <Text key={h.keys + h.label}>
            {i > 0 ? <Text> {"  "} </Text> : null}
            <Text color={COLOR.alt}>{h.keys}</Text>
            <Text color={COLOR.muted}>{` ${h.label}`}</Text>
          </Text>
        ))}
      </Text>
    </Box>
  );
}

/** Panel with the title embedded in the top border: `╭─ Title ────╮`. */
export function Panel({
  title,
  width,
  focused = false,
  height,
  children,
}: {
  title: string;
  width: number;
  focused?: boolean;
  height?: number;
  children: ReactNode;
}) {
  const color = focused ? COLOR.accent : COLOR.border;
  const w = Math.max(10, width);
  const fill = Math.max(0, w - 5 - title.length);
  return (
    <Box flexDirection="column" width={w}>
      <Box>
        <Text color={color}>{"╭─ "}</Text>
        <Text bold color={focused ? COLOR.accent : COLOR.muted}>
          {title}
        </Text>
        <Text color={color}>{` ${"─".repeat(fill)}╮`}</Text>
      </Box>
      <Box
        width={w}
        height={height}
        flexGrow={height ? 0 : 1}
        flexDirection="column"
        borderStyle="round"
        borderTop={false}
        borderColor={color}
        paddingX={1}
        overflow="hidden"
      >
        {children}
      </Box>
    </Box>
  );
}
