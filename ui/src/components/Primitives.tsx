import { useEffect, useState, type ReactNode } from "react";
import { Box, Text, useStdout } from "ink";
import { COLOR, ROLE, SPINNER_FRAMES } from "../theme.js";
import { animationsEnabled } from "../capabilities.js";

export function Rule({ width }: { width: number }) {
  return <Text color={COLOR.border}>{"─".repeat(Math.max(1, width))}</Text>;
}

export function Spinner({ label }: { label?: string }) {
  const [frame, setFrame] = useState(0);
  useEffect(() => {
    if (!animationsEnabled()) return;
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
  subtitle,
  variant = "idle",
  width,
  focused = false,
  height,
  children,
}: {
  title: string;
  subtitle?: string;
  variant?: "focus" | "idle" | "danger";
  width: number;
  focused?: boolean;
  height?: number;
  children: ReactNode;
}) {
  const active = focused || variant === "focus";
  const color = variant === "danger" ? ROLE.danger : active ? ROLE.focus : ROLE.hairline;
  const w = Math.max(12, width);
  const fill = Math.max(0, w - title.length - (subtitle ? subtitle.length + 7 : 5));
  return (
    <Box flexDirection="column" width={w}>
      <Box>
        <Text color={color}>{"╭─ "}</Text>
        <Text bold color={active ? ROLE.focus : variant === "danger" ? ROLE.danger : COLOR.muted}>
          {title}
        </Text>
        <Text color={color}>{` ${"─".repeat(fill)}`}</Text>
        {subtitle ? <Text color={COLOR.muted}>{` ${subtitle} `}</Text> : <Text color={color}>{"─"}</Text>}
        <Text color={color}>{"╮"}</Text>
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
