import { Box, Text } from "ink";
import { SLASH_COMMANDS } from "../keymap.js";
import { COLOR, lerpHex } from "../theme.js";

const CARD_BORDER = lerpHex(COLOR.alt, COLOR.border, 0.28);

const GROUPS: Array<{ title: string; hints: Array<[string, string]> }> = [
  {
    title: "Lệnh",
    hints: SLASH_COMMANDS.map(
      (command) =>
        [command.usage, command.description] as [string, string],
    ),
  },
  {
    title: "Phím",
    hints: [
      ["↵", "gửi tin nhắn / lệnh"],
      ["↑/↓", "chọn trong bảng slash command"],
      ["Tab", "điền slash command đang chọn"],
      ["?", "mở bảng phím này (khi ô nhập trống)"],
      ["bất kỳ phím", "đóng bảng này"],
      ["^c", "thoát"],
    ],
  },
];

export function HelpOverlay() {
  return (
    <Box
      flexDirection="column"
      alignSelf="center"
      borderStyle="round"
      borderColor={CARD_BORDER}
      paddingX={2}
      paddingY={1}
    >
      <Text bold color={COLOR.alt}>
        SoCa — phím & lệnh
      </Text>
      {GROUPS.map((group) => (
        <Box key={group.title} flexDirection="column" marginTop={1}>
          <Text bold color={COLOR.text}>
            {group.title}
          </Text>
          {group.hints.map(([keys, label]) => (
            <Box key={keys}>
              <Box width={24} flexShrink={0}>
                <Text color={COLOR.alt}>{keys}</Text>
              </Box>
              <Text color={COLOR.muted}>{label}</Text>
            </Box>
          ))}
        </Box>
      ))}
    </Box>
  );
}
