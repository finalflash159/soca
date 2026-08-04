import { Box, Text, useInput } from "ink";
import type { KnowledgeSetupEvent } from "../protocol.js";
import type { KnowledgeVaultStatus } from "../store.js";
import { COLOR, ICON } from "../theme.js";
import { Panel, Spinner } from "./Primitives.js";

const PHASE_LABELS: Record<string, string> = {
  scanning: "Quét vault",
  chunking: "Chia chunks",
  embedding: "Tạo embedding",
  persisting: "Ghi vector index",
  verifying: "Kiểm tra generation",
  complete: "Hoàn tất",
  failed: "Thất bại",
};

function progressBar(completed: number, total: number, width: number): string {
  if (total <= 0) return "░".repeat(width);
  const filled = Math.min(width, Math.round((completed / total) * width));
  return `${"█".repeat(filled)}${"░".repeat(width - filled)}`;
}

function boundedCount(value: number | undefined): number {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.max(0, value)
    : 0;
}

export interface KnowledgeIndexScreenProps {
  width: number;
  vault: KnowledgeVaultStatus | null;
  setup: KnowledgeSetupEvent | null;
  onReturn: () => void;
}

export function KnowledgeIndexScreen({
  width,
  vault,
  setup,
  onReturn,
}: KnowledgeIndexScreenProps) {
  const isIndexEvent = setup?.action === "index";
  const status = isIndexEvent ? setup?.status : null;
  const running = status === "running" || setup === null || !isIndexEvent;
  const failed = status === "failed";
  const busy = status === "busy";
  const phase = setup?.phase ?? (failed ? "failed" : "scanning");
  const completed = boundedCount(setup?.completed_chunks);
  const total = boundedCount(setup?.total_chunks);
  const reused = boundedCount(setup?.reused_chunks);
  const embedded = boundedCount(setup?.embedded_chunks);
  const title = running ? "Indexing Knowledge Vault" : "Knowledge index";
  const variant: "busy" | "danger" | "info" | "success" = running
    ? "busy"
    : failed
      ? "danger"
      : busy
        ? "info"
        : "success";

  useInput((_input, key) => {
    if (key.return && !running) onReturn();
  });

  return (
    <Box flexDirection="column" paddingX={1} marginBottom={1}>
      <Panel
        title={title}
        subtitle={running ? "running" : undefined}
        variant={variant}
        width={width}
      >
        <Box flexDirection="column">
          <Text color={COLOR.muted} wrap="truncate-end">
            Vault: {vault?.path ?? setup?.vault ?? "chưa xác định"}
          </Text>

          <Box marginTop={1} flexDirection="column">
            <Text
              color={running ? COLOR.accent : failed ? COLOR.bad : COLOR.good}
            >
              {running ? (
                <Spinner color={COLOR.accent} />
              ) : failed ? (
                ICON.err
              ) : (
                ICON.on
              )}
              {` ${PHASE_LABELS[phase] ?? "Cập nhật index"}`}
            </Text>
            <Text color={COLOR.text}>
              {progressBar(completed, total, Math.max(12, Math.min(42, width - 28)))}
            </Text>
            <Text color={COLOR.muted}>
              {total > 0
                ? `${completed}/${total} chunks · dùng lại ${reused} · tạo mới ${embedded}`
                : "Đang xác định số lượng chunks…"}
            </Text>
          </Box>

          <Text color={failed ? COLOR.bad : busy ? COLOR.warn : COLOR.muted}>
            {failed
              ? `${setup?.error_code ? `${setup.error_code}: ` : ""}${setup?.detail ?? "Index thất bại."}`
              : busy
                ? setup?.detail ?? "Một tiến trình index khác đang chạy."
                : running
                  ? setup?.detail ?? "Đang chuẩn bị…"
                  : setup?.detail ?? "Đã index xong."}
          </Text>

          {!running ? (
            <Box marginTop={1} flexDirection="column">
              <Text color={COLOR.text}>
                {setup?.documents ?? 0} tài liệu · {setup?.chunks ?? total} chunks
              </Text>
              <Text color={COLOR.muted}>
                revision {setup?.revision ?? "?"} · dense {setup?.dense_state ?? "unknown"}
              </Text>
              <Text color={COLOR.alt}>Enter quay lại Settings</Text>
            </Box>
          ) : null}
        </Box>
      </Panel>
    </Box>
  );
}
