import { Box, Text } from "ink";
import { COLOR, rampColor } from "../theme.js";

// Perched songbird mark; catches the dawn light top-left to bottom-right.
export const BIRD_LINES: readonly string[] = [
  "   .-.",
  " .(o/)>",
  " ///_/",
  "  //'",
];

export function Bird() {
  const rows = Math.max(1, BIRD_LINES.length - 1);
  return (
    <Box flexDirection="column">
      {BIRD_LINES.map((line, row) => {
        const chars = [...line];
        const cols = Math.max(1, chars.length - 1);
        return (
          <Box key={row}>
            {chars.map((ch, col) =>
              ch === " " ? (
                <Text key={col}> </Text>
              ) : (
                <Text
                  key={col}
                  bold
                  color={rampColor((col / cols + row / rows) / 2)}
                >
                  {ch}
                </Text>
              ),
            )}
          </Box>
        );
      })}
    </Box>
  );
}

export function Wordmark() {
  const word = "SoCa";
  const last = word.length - 1;
  return (
    <Text>
      {[...word].map((ch, i) => (
        <Text key={i} bold color={rampColor(i / last)}>
          {ch}
        </Text>
      ))}
    </Text>
  );
}

export function InlineMark({ singing = false }: { singing?: boolean }) {
  return (
    <Text>
      <Text bold color={COLOR.accent}>
        {singing ? "(o>" : "(.>"}
      </Text>
      <Text> </Text>
      <Wordmark />
    </Text>
  );
}
