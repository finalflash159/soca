import { Box, Text } from "ink";
import { COLOR, ICON, lerpHex } from "../theme.js";
import type { TimelineEntry } from "../store.js";

/** One finished conversation line; rendered once inside <Static>. */
export function TimelineLine({ entry }: { entry: TimelineEntry }) {
  switch (entry.kind) {
    case "user":
      return (
        <Box paddingX={1} marginBottom={1}>
          <Text>
            <Text bold color={COLOR.alt}>
              {ICON.pointer}{" "}
            </Text>
            <Text color={COLOR.text}>{entry.text}</Text>
          </Text>
        </Box>
      );
    case "soca":
      return (
        <Box paddingX={1} marginBottom={1}>
          <Box
            width="100%"
            flexDirection="column"
            borderStyle="round"
            borderColor={lerpHex(COLOR.accent, COLOR.border, 0.72)}
            paddingX={1}
          >
            <Text>
              <Text bold color={COLOR.accent}>
                {ICON.bird}{" "}
              </Text>
              <Text color={COLOR.text}>{entry.text}</Text>
            </Text>
            {entry.citations && entry.citations.length > 0 ? (
              <Box flexDirection="column" marginTop={1}>
                <Text color={COLOR.border}>── nguồn</Text>
                {entry.citations.map((citation) => {
                  const lines =
                    citation.line_start && citation.line_end
                      ? `:${citation.line_start}-${citation.line_end}`
                      : "";
                  return (
                    <Text
                      key={`${citation.source}:${citation.label}:${citation.path}`}
                      color={COLOR.muted}
                    >
                      <Text bold color={COLOR.alt}>
                        {citation.label}
                      </Text>
                      {`  ${citation.title} · ${citation.path}${lines}`}
                    </Text>
                  );
                })}
              </Box>
            ) : null}
          </Box>
        </Box>
      );
    case "error":
      return (
        <Box paddingX={1}>
          <Text color={COLOR.bad}>
            {ICON.err} {entry.text}
          </Text>
        </Box>
      );
    default:
      return (
        <Box paddingX={1}>
          <Text color={COLOR.muted}>
            {ICON.dot} {entry.text}
          </Text>
        </Box>
      );
  }
}
