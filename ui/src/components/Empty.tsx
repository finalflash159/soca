import { Box, Text } from "ink";
import { COLOR } from "../theme.js";

export function Empty({
  icon,
  title,
  hint,
}: {
  icon: string;
  title: string;
  hint: string;
}) {
  return (
    <Box flexDirection="column" paddingX={1} paddingY={1}>
      <Text color={COLOR.muted}>
        {icon} {title}
      </Text>
      <Text color={COLOR.border}>{`  ${hint}`}</Text>
    </Box>
  );
}
