import { Box, Text } from "ink";
import { COLOR, ICON, ROLE } from "../theme.js";
import { FooterHints, type Hint } from "./Primitives.js";

export function StatusBar({
  hints,
  mode,
  profile,
  memoryOn,
  llm,
  remote,
}: {
  hints: Hint[];
  mode: string;
  profile: string;
  memoryOn: boolean;
  llm: string;
  remote: boolean;
}) {
  return (
    <Box justifyContent="space-between">
      <FooterHints hints={hints} />
      <Box paddingX={1}>
        <Text color={COLOR.muted}>
          <Text bold color={ROLE.focus}>
            {mode}
          </Text>
          {` ${ICON.dot} ${profile} ${ICON.dot} mem${memoryOn ? ICON.on : ICON.off}`}
          {llm ? (
            <Text>
              {` ${ICON.dot} `}
              <Text color={remote ? ROLE.busy : ROLE.ok}>
                {remote ? `remote ${llm}` : `local ${llm}`}
              </Text>
            </Text>
          ) : null}
        </Text>
      </Box>
    </Box>
  );
}
