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
