import { Box, Text } from "ink";
import type {
  TurnProgressEvent,
  TurnProgressPhase,
} from "../protocol.js";
import { COLOR, ICON, ROLE } from "../theme.js";
import { Panel, Spinner, type PanelVariant } from "./Primitives.js";

interface PhaseView {
  label: string;
  detail: string;
  color: string;
  variant: PanelVariant;
}

const PHASE_VIEW: Record<TurnProgressPhase, PhaseView> = {
  preparing: {
    label: "Chuẩn bị",
    detail: "khởi tạo runtime, model và index",
    color: ROLE.info,
    variant: "info",
  },
  analyzing: {
    label: "Phân tích yêu cầu",
    detail: "chuẩn hóa đầu vào và kiểm tra an toàn",
    color: ROLE.focus,
    variant: "focus",
  },
  routing: {
    label: "Định tuyến",
    detail: "chọn hội thoại, memory, knowledge hoặc tool",
    color: ROLE.info,
    variant: "info",
  },
  memory: {
    label: "Truy xuất memory",
    detail: "lấy context làm việc và ghi nhớ liên quan",
    color: ROLE.ok,
    variant: "success",
  },
  retrieval: {
    label: "Tra cứu knowledge",
    detail: "tìm, xếp hạng và chọn bằng chứng",
    color: COLOR.accentBright,
    variant: "focus",
  },
  tool: {
    label: "Thực thi tool",
    detail: "gọi capability đã được router chọn",
    color: ROLE.busy,
    variant: "busy",
  },
  synthesis: {
    label: "Tổng hợp câu trả lời",
    detail: "đưa context vào LLM và tạo phản hồi",
    color: ROLE.busy,
    variant: "busy",
  },
  validation: {
    label: "Kiểm tra câu trả lời",
    detail: "kiểm tra grounding, citation và an toàn",
    color: ROLE.ok,
    variant: "success",
  },
  speech: {
    label: "Tổng hợp giọng nói",
    detail: "tạo và phát audio phản hồi",
    color: ROLE.ok,
    variant: "success",
  },
  complete: {
    label: "Hoàn tất",
    detail: "",
    color: ROLE.ok,
    variant: "success",
  },
};

function operationDetail(event: TurnProgressEvent, fallback: string): string {
  const toolPrefix = "tool:";
  if (event.operation.startsWith(toolPrefix)) {
    return `đang chạy ${event.operation.slice(toolPrefix.length)}`;
  }
  if (event.operation === "speech_recognition") {
    return "chuyển âm thanh thành văn bản";
  }
  return fallback;
}

export function TurnProgress({
  progress,
  completed,
  width,
}: {
  progress: TurnProgressEvent;
  completed: TurnProgressPhase[];
  width: number;
}) {
  const view = PHASE_VIEW[progress.phase];
  const visibleCompleted = completed.slice(-4);
  return (
    <Box paddingX={1} marginBottom={1}>
      <Panel
        title="SoCa đang xử lý"
        subtitle={progress.surface === "voice" ? "voice" : "chat"}
        width={width}
        variant={view.variant}
      >
        <Spinner
          label={view.label}
          color={view.color}
          labelColor={view.color}
        />
        <Text color={COLOR.muted}>
          {operationDetail(progress, view.detail)}
        </Text>
        {visibleCompleted.length > 0 ? (
          <Box marginTop={1}>
            <Text color={COLOR.muted}>
              {visibleCompleted.map((phase, index) => (
                <Text key={phase}>
                  {index > 0 ? `  ${ICON.dot}  ` : ""}
                  <Text color={ROLE.ok}>{ICON.ok}</Text>{" "}
                  {PHASE_VIEW[phase].label}
                </Text>
              ))}
            </Text>
          </Box>
        ) : null}
      </Panel>
    </Box>
  );
}
