import { Box, Text } from "ink";
import type { SessionMemoryStats } from "../protocol.js";
import { COLOR, ROLE } from "../theme.js";

function compactTokens(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}m`;
  if (value < 1_000) return String(value);
  const digits = value >= 10_000 ? 1 : 2;
  return `${(value / 1_000).toFixed(digits)}k`;
}

export function SessionTokenMeter({
  stats,
  width = 12,
}: {
  stats: SessionMemoryStats | null;
  width?: number;
}) {
  if (stats === null) {
    return (
      <Box paddingX={1}>
        <Text color={COLOR.muted}>session memory tắt</Text>
      </Box>
    );
  }
  const ratio = Math.max(
    0,
    Math.min(1, stats.current_tokens / stats.hard_limit_tokens),
  );
  const filled = Math.round(ratio * width);
  const color =
    stats.current_tokens >= stats.high_watermark_tokens
      ? ROLE.danger
      : ratio >= 0.75
        ? ROLE.busy
        : ROLE.ok;
  return (
    <Box paddingX={1} justifyContent="flex-end">
      <Text color={COLOR.muted}>session </Text>
      <Text color={color}>{"█".repeat(filled)}</Text>
      <Text color={COLOR.border}>{"░".repeat(Math.max(0, width - filled))}</Text>
      <Text color={COLOR.muted}>
        {` ~${compactTokens(stats.current_tokens)} / ${compactTokens(stats.hard_limit_tokens)} tok`}
      </Text>
    </Box>
  );
}

export { compactTokens };
