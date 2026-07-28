import { Box, Text } from "ink";
import type { SlashCommandDefinition } from "../keymap.js";
import { COLOR, ICON, ROLE } from "../theme.js";
import { Panel } from "./Primitives.js";

export function CommandPalette({
  commands,
  selectedIndex,
  width,
}: {
  commands: SlashCommandDefinition[];
  selectedIndex: number;
  width: number;
}) {
  const selected = Math.min(selectedIndex, Math.max(0, commands.length - 1));
  return (
    <Box paddingX={1} marginBottom={1}>
      <Panel
        title="slash commands"
        subtitle={`${commands.length} match`}
        width={width}
        variant="focus"
      >
        {commands.length === 0 ? (
          <Text color={COLOR.muted}>
            {ICON.dot} Không có lệnh khớp. Tiếp tục gõ hoặc Esc để đóng.
          </Text>
        ) : (
          commands.map((command, index) => (
            <Box key={command.value}>
              <Box width={30} flexShrink={0}>
                <Text
                  bold={index === selected}
                  color={index === selected ? ROLE.focus : COLOR.alt}
                >
                  {index === selected ? `${ICON.pointer} ` : "  "}
                  {command.usage}
                </Text>
              </Box>
              <Text color={COLOR.muted} wrap="truncate-end">
                {command.description}
              </Text>
            </Box>
          ))
        )}
        <Text color={COLOR.muted}>↑/↓ chọn · Tab điền · Enter chạy · Esc đóng</Text>
      </Panel>
    </Box>
  );
}
