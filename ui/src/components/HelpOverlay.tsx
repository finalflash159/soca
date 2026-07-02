import { Box, Text } from "ink";
import { COLOR, lerpHex } from "../theme.js";

const CARD_BORDER = lerpHex(COLOR.accent, COLOR.border, 0.55);

const GROUPS: Array<{ title: string; hints: Array<[string, string]> }> = [
  {
    title: "Lệnh",
    hints: [
      ["/chat /voice /status", "chuyển chế độ"],
      ["/listen", "chạy voice loop"],
      ["/stop", "dừng voice loop"],
      ["/quit", "thoát"],
    ],
  },
  {
    title: "Phím",
    hints: [
      ["↵", "gửi tin nhắn / lệnh"],
      ["?", "mở/đóng bảng phím (khi ô nhập trống)"],
      ["esc", "đóng bảng này"],
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
      <Text bold color={COLOR.accent}>
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
