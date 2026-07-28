import { Box, Text } from "ink";
import type {
  ContextEvent,
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
import { Panel } from "./Primitives.js";

function valueOrDynamic(value: number | null): string {
  return value === null ? "theo câu hỏi" : `~${compactTokens(value)} tok`;
}

function boundedText(value: string, limit: number): string {
  const normalized = value.trim();
  if (normalized.length <= limit) return normalized;
  return `…${normalized.slice(-(limit - 1))}`;
}

function ContextBody({ context }: { context: ContextEvent | null }) {
  if (context === null)
    return <Text color={COLOR.muted}>đang lấy context breakdown…</Text>;
  return (
    <Box flexDirection="column">
      {context.components.map((component) => (
        <Box key={component.id}>
          <Box width={34} flexShrink={0}>
            <Text color={COLOR.text}>{component.label}</Text>
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
      <Text color={COLOR.muted}>
        ~ là ước lượng UTF-8/4; knowledge, archive memory và câu hỏi chỉ được tính
        khi có lượt cụ thể.
      </Text>
    </Box>
  );
}

function MemoryBody({
  memory,
  compaction,
}: {
  memory: MemoryEvent | null;
  compaction: string;
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
      {compaction ? (
        <Text color={COLOR.alt}>compact: {compaction}</Text>
      ) : null}
      <Box marginTop={1} flexDirection="column">
        <Text bold color={ROLE.focus}>Working summary</Text>
        <Text color={memory.summary ? COLOR.text : COLOR.muted}>
          {memory.summary
            ? boundedText(memory.summary, 1_200)
            : "(chưa có summary)"}
        </Text>
      </Box>
      <Box marginTop={1} flexDirection="column">
        <Text bold color={ROLE.focus}>Recent conversation</Text>
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

function UsageBody({ usage }: { usage: UsageEvent | null }) {
  if (usage === null)
    return <Text color={COLOR.muted}>đang lấy usage…</Text>;
  return (
    <Box flexDirection="column">
      <Text>{usage.turns} lượt · {usage.llm_turns} lượt gọi LLM</Text>
      <Text>
        prompt đã gửi {usage.prompt_tokens} tok · completion{" "}
        {usage.completion_tokens} tok
      </Text>
      <Text>
        TTFT trung bình ~{Math.round(usage.mean_ttft_ms)} ms ·{" "}
        {usage.mean_tokens_per_second.toFixed(1)} tok/s
      </Text>
      <Text color={COLOR.muted}>
        Đây là usage tích lũy của provider/model, không phải dung lượng context
        hiện đang giữ. Dùng /context cho context hiện tại.
      </Text>
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
  memoryCompaction: string;
}) {
  const titles: Record<InfoView, string> = {
    status: "runtime status",
    context: "context breakdown",
    memory: "working memory",
    usage: "session usage",
  };
  return (
    <Box paddingX={1} marginBottom={1}>
      <Panel title={titles[view]} subtitle="tạm thời" width={width}>
        {view === "context" ? <ContextBody context={context} /> : null}
        {view === "memory" ? (
          <MemoryBody memory={memory} compaction={memoryCompaction} />
        ) : null}
        {view === "status" ? (
          <StatusBody profiles={profiles} knowledge={knowledge} />
        ) : null}
        {view === "usage" ? <UsageBody usage={usage} /> : null}
        <Box marginTop={1}>
          <Text color={COLOR.muted}>
            {ICON.dot} bắt đầu nhập để đóng và trở lại cuộc trò chuyện
          </Text>
        </Box>
      </Panel>
    </Box>
  );
}
