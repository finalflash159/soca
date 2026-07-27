import { Box, Text } from "ink";
import { COLOR, lerpHex } from "../theme.js";

const CARD_BORDER = lerpHex(COLOR.accent, COLOR.border, 0.55);

const GROUPS: Array<{ title: string; hints: Array<[string, string]> }> = [
  {
    title: "Lệnh",
    hints: [
      ["/chat /voice /status /settings", "chuyển chế độ"],
      ["/s", "mở cài đặt LLM"],
      ["/listen", "chạy voice loop"],
      ["/stop", "dừng voice loop"],
      ["/k <câu hỏi>", "ép dùng knowledge context"],
      ["/memory", "xem session memory"],
      ["/usage", "token / latency của phiên"],
      ["/quit", "thoát"],
    ],
  },
  {
    title: "Phím",
    hints: [
      ["↵", "gửi tin nhắn / lệnh"],
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
