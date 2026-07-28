import { Box, Text } from "ink";
import type {
  ContextEvent,
  MemoryCompactionEvent,
  MemoryEvent,
  UsageEvent,
} from "../protocol.js";
import type {
  InfoView,
  KnowledgeIndexStatus,
  StatusProfile,
} from "../store.js";
import { COLOR, ICON, ROLE } from "../theme.js";
import { compactTokens } from "./SessionTokenMeter.js";
import { Panel, Spinner, type PanelVariant } from "./Primitives.js";

function valueOrDynamic(value: number | null): string {
  return value === null ? "theo câu hỏi" : `~${compactTokens(value)} tok`;
}

function boundedText(value: string, limit: number): string {
  const normalized = value.trim();
  if (normalized.length <= limit) return normalized;
  return `…${normalized.slice(-(limit - 1))}`;
}

function contextColor(id: ContextEvent["components"][number]["id"]): string {
  if (id === "system") return COLOR.accent;
  if (id === "core_memory") return COLOR.alt;
  if (id === "working_summary") return COLOR.good;
  if (id === "recent_conversation") return COLOR.accentBright;
  return COLOR.text;
}

function formatLatency(milliseconds: number): string {
  if (milliseconds >= 1_000) return `${(milliseconds / 1_000).toFixed(2)} s`;
  return `${Math.round(milliseconds)} ms`;
}

function SessionUsageSection({ usage }: { usage: UsageEvent | null }) {
  return (
    <Box marginTop={1} flexDirection="column">
      <Text bold color={ROLE.busy}>
        Usage LLM tích lũy
      </Text>
      {usage === null ? (
        <Text color={COLOR.muted}>đang lấy usage…</Text>
      ) : (
        <>
          <Box>
            <Box width={34} flexShrink={0}>
              <Text color={COLOR.text}>Hoạt động phiên</Text>
            </Box>
            <Text color={ROLE.busy}>
              {usage.turns} lượt hội thoại · {usage.llm_turns} lần gọi LLM
            </Text>
          </Box>
          <Box>
            <Box width={34} flexShrink={0}>
              <Text color={COLOR.text}>Token provider</Text>
            </Box>
            <Text color={COLOR.accentBright}>
              {usage.prompt_tokens} prompt · {usage.completion_tokens} completion ·{" "}
              {usage.prompt_tokens + usage.completion_tokens} total
            </Text>
          </Box>
          <Box>
            <Box width={34} flexShrink={0}>
              <Text color={COLOR.text}>Hiệu năng trung bình</Text>
            </Box>
            <Text color={COLOR.alt}>
              TTFT {formatLatency(usage.mean_ttft_ms)} ·{" "}
              {usage.mean_tokens_per_second.toFixed(1)} tok/s
            </Text>
          </Box>
        </>
      )}
    </Box>
  );
}

function ContextBody({
  context,
  usage,
}: {
  context: ContextEvent | null;
  usage: UsageEvent | null;
}) {
  if (context === null)
    return <Text color={COLOR.muted}>đang lấy context breakdown…</Text>;
  return (
    <Box flexDirection="column">
      {context.components.map((component) => (
        <Box key={component.id}>
          <Box width={34} flexShrink={0}>
            <Text color={contextColor(component.id)}>{component.label}</Text>
          </Box>
          <Box width={18} flexShrink={0}>
            <Text color={component.tokens === null ? COLOR.alt : ROLE.focus}>
              {valueOrDynamic(component.tokens)}
            </Text>
          </Box>
          <Text color={COLOR.muted}>{component.policy}</Text>
        </Box>
      ))}
      <Box marginTop={1} flexDirection="column">
        <Text color={COLOR.text}>
          prompt nền hiện tại: ~{compactTokens(context.resident_prompt_tokens)} tok
        </Text>
        <Text color={COLOR.text}>
          output reserve: {compactTokens(context.output_reserve_tokens)} tok
        </Text>
        <Text color={COLOR.text}>
          model window:{" "}
          {context.model_context_tokens === null
            ? "chưa biết"
            : `${compactTokens(context.model_context_tokens)} tok`}
        </Text>
        <Text color={COLOR.text}>
          còn cho input + retrieval:{" "}
          {context.available_dynamic_tokens === null
            ? "chưa tính được"
            : `~${compactTokens(context.available_dynamic_tokens)} tok`}
        </Text>
      </Box>
      <SessionUsageSection usage={usage} />
    </Box>
  );
}

function MemoryBody({
  memory,
}: {
  memory: MemoryEvent | null;
}) {
  if (memory === null)
    return <Text color={COLOR.muted}>đang lấy working memory…</Text>;
  if (!memory.enabled)
    return <Text color={COLOR.muted}>session memory đang tắt.</Text>;
  const stats = memory.stats;
  return (
    <Box flexDirection="column">
      {stats ? (
        <>
          <Text color={COLOR.text}>
            {`~${compactTokens(stats.current_tokens)} / ${compactTokens(stats.hard_limit_tokens)} tok · compact tại ${compactTokens(stats.high_watermark_tokens)} · target ${compactTokens(stats.target_tokens)}`}
          </Text>
          <Text color={COLOR.muted}>
            {`${stats.complete_turn_count}/${stats.turn_count} turn hoàn chỉnh · summary ${stats.summary_tokens} tok · recent ${stats.recent_tokens} tok · worker ${stats.worker_state}`}
          </Text>
        </>
      ) : null}
      <Box marginTop={1} flexDirection="column">
        <Text bold color={ROLE.ok}>Working summary</Text>
        <Text color={memory.summary ? COLOR.text : COLOR.muted}>
          {memory.summary
            ? boundedText(memory.summary, 1_200)
            : "(chưa có summary)"}
        </Text>
      </Box>
      <Box marginTop={1} flexDirection="column">
        <Text bold color={COLOR.accentBright}>Recent conversation</Text>
        <Text color={memory.recent ? COLOR.text : COLOR.muted}>
          {memory.recent
            ? boundedText(memory.recent, 1_800)
            : "(chưa có turn)"}
        </Text>
      </Box>
      <Text color={COLOR.muted}>
        Archive/core memory không nằm ở đây; dùng /context để xem phần prompt và
        /memory proposals để duyệt ghi nhớ dài hạn.
      </Text>
    </Box>
  );
}

const COMPACTION_ACTIVE = new Set(["accepted", "running"]);

function compactionDetail(compaction: MemoryCompactionEvent): string {
  const detail = compaction.detail;
  if (detail === "not_enough_complete_turns") {
    const minimum = compaction.minimum_complete_turns ?? 5;
    return `Cần ít nhất ${minimum} lượt hoàn chỉnh để manual compact; hiện có ${compaction.complete_turns ?? 0}/${minimum}.`;
  }
  const labels: Record<string, string> = {
    below_compaction_boundary: "Working memory chưa đạt ngưỡng compact tự động.",
    compaction_already_running: "Một lượt compact khác đang chạy.",
    summary_model_not_configured: "Chưa cấu hình model summary local.",
    summary_model_not_provisioned:
      "Weight của model summary chưa có hoặc không đạt kiểm tra permission.",
    worker_timeout: "Summary worker đã vượt timeout.",
    worker_exited_without_payload: "Summary worker dừng mà không trả kết quả.",
    invalid_worker_payload: "Summary worker trả payload không hợp lệ.",
    invalid_summary_artifact: "Summary không đạt schema an toàn.",
    empty_continuity_summary:
      "Model trả working summary rỗng; lịch sử gốc được giữ nguyên.",
  };
  return detail ? (labels[detail] ?? detail) : "";
}

function CompactionBody({
  compaction,
}: {
  compaction: MemoryCompactionEvent | null;
}) {
  if (compaction === null) {
    return (
      <Text color={COLOR.muted}>
        Chưa có lượt compact trong phiên này. Dùng /memory compact để bắt đầu.
      </Text>
    );
  }
  const before = compaction.before_tokens;
  const after = compaction.after_tokens;
  const elapsed =
    compaction.elapsed_ms === null || compaction.elapsed_ms === undefined
      ? null
      : compaction.elapsed_ms / 1_000;
  if (COMPACTION_ACTIVE.has(compaction.status)) {
    return (
      <Box flexDirection="column">
        <Spinner
          label={`đang compact${compaction.generation ? ` · generation ${compaction.generation}` : ""}`}
        />
        <Text color={COLOR.muted}>
          {before === null || before === undefined
            ? "đang chuẩn bị working memory…"
            : `nguồn ~${compactTokens(before)} token · ${compaction.compacted_turns ?? 0} turn cũ`}
          {elapsed === null ? "" : ` · ${elapsed.toFixed(1)}s`}
        </Text>
        <Text color={COLOR.alt}>
          Model summary chạy trong worker riêng và sẽ tự shutdown khi hoàn tất.
        </Text>
      </Box>
    );
  }
  if (compaction.status === "published") {
    const saved =
      before !== null &&
      before !== undefined &&
      after !== null &&
      after !== undefined
        ? before - after
        : null;
    return (
      <Box flexDirection="column">
        <Text bold color={ROLE.ok}>
          {ICON.ok} Compact hoàn tất
        </Text>
        <Text color={COLOR.text}>
          {before === null ||
          before === undefined ||
          after === null ||
          after === undefined
            ? "Token metrics không khả dụng."
            : `~${compactTokens(before)} → ~${compactTokens(after)} token`}
          {saved === null
            ? ""
            : saved >= 0
              ? ` · giảm ~${compactTokens(saved)}`
              : ` · tăng ~${compactTokens(Math.abs(saved))}`}
        </Text>
        <Text color={COLOR.muted}>
          {`${compaction.compacted_turns ?? 0} turn cũ đã được thay bằng working summary`}
          {elapsed === null ? "" : ` · ${elapsed.toFixed(1)}s`}
        </Text>
        <Text color={COLOR.alt}>Xem nội dung: /memory compact show</Text>
      </Box>
    );
  }
  if (compaction.status === "cancelled") {
    return (
      <Text color={COLOR.alt}>
        {ICON.dot} Đã hủy compact; working memory cũ được giữ nguyên.
      </Text>
    );
  }
  const failed = ["failed", "unavailable"].includes(compaction.status);
  return (
    <Box flexDirection="column">
      <Text color={failed ? ROLE.danger : ROLE.busy}>
        {failed ? ICON.err : ICON.dot} Compact: {compaction.status}
      </Text>
      <Text color={COLOR.muted}>{compactionDetail(compaction)}</Text>
    </Box>
  );
}

function CompactedSummaryBody({ memory }: { memory: MemoryEvent | null }) {
  if (memory === null)
    return <Text color={COLOR.muted}>đang lấy working summary…</Text>;
  if (!memory.enabled)
    return <Text color={COLOR.muted}>session memory đang tắt.</Text>;
  if (!memory.summary) {
    return (
      <Box flexDirection="column">
        <Text color={COLOR.muted}>Chưa có working summary đã compact.</Text>
        <Text color={COLOR.alt}>Dùng /memory compact để tạo thủ công.</Text>
      </Box>
    );
  }
  return (
    <Box flexDirection="column">
      <Text color={COLOR.muted}>
        generation {memory.stats?.summary_generation ?? "?"} · ~
        {compactTokens(memory.stats?.summary_tokens ?? 0)} token
      </Text>
      <Box marginTop={1}>
        <Text color={COLOR.text}>{memory.summary}</Text>
      </Box>
    </Box>
  );
}

function StatusBody({
  profiles,
  knowledge,
}: {
  profiles: StatusProfile[];
  knowledge: KnowledgeIndexStatus | null;
}) {
  if (profiles.length === 0)
    return <Text color={COLOR.muted}>đang quét profile…</Text>;
  return (
    <Box flexDirection="column">
      {profiles.map((profile) => (
        <Box key={profile.key}>
          <Box width={18} flexShrink={0}>
            <Text bold color={COLOR.alt}>{profile.key}</Text>
          </Box>
          <Box width={9} flexShrink={0}>
            <Text color={profile.status === "ok" ? ROLE.ok : ROLE.busy}>
              {profile.status}
            </Text>
          </Box>
          <Text color={COLOR.muted} wrap="truncate-end">
            {profile.asr} {ICON.dot} {profile.llm} {ICON.dot} {profile.tts}
            {profile.voice ? `/${profile.voice}` : ""}
          </Text>
        </Box>
      ))}
      {knowledge ? (
        <Text color={COLOR.muted}>
          knowledge · {knowledge.sparse_state} · dense {knowledge.dense_state} ·{" "}
          {knowledge.documents} docs / {knowledge.chunks} chunks
        </Text>
      ) : null}
    </Box>
  );
}

export function InformationPanel({
  view,
  width,
  context,
  memory,
  usage,
  profiles,
  knowledge,
  memoryCompaction,
}: {
  view: InfoView;
  width: number;
  context: ContextEvent | null;
  memory: MemoryEvent | null;
  usage: UsageEvent | null;
  profiles: StatusProfile[];
  knowledge: KnowledgeIndexStatus | null;
  memoryCompaction: MemoryCompactionEvent | null;
}) {
  const titles: Record<InfoView, string> = {
    status: "runtime status",
    context: "context & usage",
    memory: "working memory",
    compaction: "memory compaction",
    compacted_summary: "compacted summary",
  };
  const variants: Record<InfoView, PanelVariant> = {
    status: "info",
    context: "focus",
    memory: "success",
    compaction:
      memoryCompaction === null || memoryCompaction.status === "idle"
        ? "info"
        : COMPACTION_ACTIVE.has(memoryCompaction.status)
          ? "busy"
          : memoryCompaction.status === "published"
            ? "success"
            : ["failed", "unavailable"].includes(memoryCompaction.status)
              ? "danger"
              : "info",
    compacted_summary: "success",
  };
  return (
    <Box paddingX={1} marginBottom={1}>
      <Panel
        title={titles[view]}
        subtitle={
          view === "compaction" && memoryCompaction
            ? memoryCompaction.status
            : view === "compacted_summary"
              ? "expanded"
              : undefined
        }
        width={width}
        variant={variants[view]}
      >
        {view === "context" ? (
          <ContextBody context={context} usage={usage} />
        ) : null}
        {view === "memory" ? <MemoryBody memory={memory} /> : null}
        {view === "status" ? (
          <StatusBody profiles={profiles} knowledge={knowledge} />
        ) : null}
        {view === "compaction" ? (
          <CompactionBody compaction={memoryCompaction} />
        ) : null}
        {view === "compacted_summary" ? (
          <CompactedSummaryBody memory={memory} />
        ) : null}
      </Panel>
    </Box>
  );
}
