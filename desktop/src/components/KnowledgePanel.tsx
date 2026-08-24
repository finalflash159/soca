/** Knowledge source, retrieval index, session memory, and last retrieval. */

import { BookOpen, Database, FolderOpen, Search } from "lucide-react";

import { EmptyState, Field, Section, Stat } from "@/components/Page";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { IndexJob, KnowledgeState } from "@/engine/knowledge";
import { evidenceSummary, indexJobRunning, memoryModeSummary } from "@/engine/knowledge";
import { cn } from "@/lib/utils";

interface KnowledgePanelProps {
  knowledge: KnowledgeState;
  connected: boolean;
  onInit: () => void;
  onIndex: () => void;
  onRefreshMemory: () => void;
  onCompact: () => void;
  onApprove: (id: string) => void;
  onReject: (id: string) => void;
}

function score(value: number | null | undefined): string {
  return typeof value === "number" ? value.toFixed(3) : "—";
}

const PHASE_LABEL: Record<string, string> = {
  scanning: "Đang quét tài liệu",
  chunking: "Đang chia chunk",
  embedding: "Đang tạo embedding",
  persisting: "Đang ghi vector index",
  verifying: "Đang kiểm tra",
  complete: "Đã xong",
};

/**
 * What the build is doing, with a bar that actually moves.
 *
 * The engine sends `completed_chunks` and `total_chunks` on every step. A
 * spinner alone cannot distinguish a long embedding pass from a wedged thread,
 * which is why this screen used to read as hung.
 */
function IndexProgress({ job }: { job: IndexJob }) {
  const done = job.completedChunks;
  const total = job.totalChunks;
  const fraction = total !== null && total > 0 && done !== null ? Math.min(1, done / total) : null;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between gap-3 text-sm">
        <span>{job.phase !== null ? (PHASE_LABEL[job.phase] ?? job.detail) : job.detail}</span>
        {done !== null && total !== null && total > 0 && (
          <span className="text-muted-foreground font-mono text-xs tabular-nums">
            {done}/{total}
          </span>
        )}
      </div>
      <div className="bg-secondary h-1.5 w-full overflow-hidden rounded-full">
        <div
          className={cn(
            "bg-primary h-full rounded-full",
            // With no total yet — the scanning phase — a fixed sliver that
            // pulses says "working" without claiming a position it cannot know.
            fraction === null ? "w-1/4 animate-pulse" : "transition-[width] duration-300",
          )}
          style={fraction === null ? undefined : { width: `${fraction * 100}%` }}
        />
      </div>
      {(job.reusedChunks ?? 0) > 0 && (
        <p className="text-muted-foreground text-xs">
          Dùng lại {job.reusedChunks} chunk chưa đổi · nhúng mới {job.embeddedChunks ?? 0}
        </p>
      )}
    </div>
  );
}

function VaultSection({
  knowledge,
  connected,
  onInit,
}: Pick<KnowledgePanelProps, "knowledge" | "connected" | "onInit">) {
  const running = indexJobRunning(knowledge.indexJob);
  const vault = knowledge.vault;
  const initialized = vault?.initialized === true;

  return (
    <Section
      icon={FolderOpen}
      title="Knowledge source"
      description="Markdown files SoCa may retrieve from."
      actions={
        // Init only when there is something to initialise. Offering it on a
        // ready vault is what made the button unexplainable.
        !initialized ? (
          <Button size="sm" disabled={!connected || running} onClick={onInit}>
            Create structure
          </Button>
        ) : null
      }
    >
      {vault === null ? (
        <p className="text-muted-foreground text-sm">Chưa nhận được trạng thái vault từ engine.</p>
      ) : (
        <div className="flex flex-col gap-3">
          <Field label="Source folder">
            <div className="border-border bg-muted/40 flex h-10 items-center rounded-lg border px-3">
              <span className="truncate font-mono text-xs">{vault.path}</span>
            </div>
          </Field>
          {!initialized && (
            <p className="text-muted-foreground text-sm leading-6">
              Create the folder structure before indexing. Existing files are kept.
            </p>
          )}
        </div>
      )}
    </Section>
  );
}

function IndexSection({
  knowledge,
  connected,
  onIndex,
}: Pick<KnowledgePanelProps, "knowledge" | "connected" | "onIndex">) {
  const running = indexJobRunning(knowledge.indexJob);
  const index = knowledge.index;
  const job = knowledge.indexJob;
  const initialized = knowledge.vault?.initialized === true;

  return (
    <Section
      icon={Database}
      title="Chỉ mục truy xuất"
      description="Không có chỉ mục thì trợ lý không tìm được gì trong vault."
      actions={
        <Button
          size="sm"
          variant={index === null ? "default" : "outline"}
          disabled={!connected || running || !initialized}
          onClick={onIndex}
        >
          {running ? "Đang dựng…" : index === null ? "Dựng chỉ mục" : "Dựng lại"}
        </Button>
      }
    >
      {running && job !== null ? (
        <IndexProgress job={job} />
      ) : index === null ? (
        <p className="text-muted-foreground text-sm leading-6">
          {initialized
            ? "Chưa có chỉ mục. Dựng một lần, rồi chỉ dựng lại khi tài liệu đổi."
            : "Tạo vault trước đã."}
        </p>
      ) : (
        <div className="flex flex-col gap-2">
          <Stat label="Tài liệu">{index.documents}</Stat>
          <Stat label="Chunk">{index.chunks}</Stat>
          <Stat label="Tìm theo từ khoá">
            <Badge variant={index.sparseState === "ready" ? "secondary" : "outline"}>
              {index.sparseState}
            </Badge>
          </Stat>
          <Stat label="Tìm theo ngữ nghĩa">
            <Badge variant={index.denseState === "ready" ? "secondary" : "outline"}>
              {index.denseState}
            </Badge>
          </Stat>
        </div>
      )}

      {job?.errorCode != null && !running && (
        <p className="text-destructive text-sm">
          {job.detail} ({job.errorCode})
        </p>
      )}
    </Section>
  );
}

function MemorySection({
  knowledge,
  connected,
  onRefreshMemory,
  onCompact,
  onApprove,
  onReject,
}: Omit<KnowledgePanelProps, "onInit" | "onIndex">) {
  const trace = knowledge.memoryTrace;
  const proposals = knowledge.proposals;

  return (
    <Section
      icon={BookOpen}
      title="Bộ nhớ"
      description={memoryModeSummary(trace)}
      actions={
        <>
          <Button size="sm" variant="ghost" disabled={!connected} onClick={onRefreshMemory}>
            Tải lại
          </Button>
          <Button size="sm" variant="outline" disabled={!connected} onClick={onCompact}>
            Nén
          </Button>
        </>
      }
    >
      {trace !== null && (
        <div className="flex flex-col gap-2">
          {trace.recentTurnCount !== null && (
            <Stat label="Lượt gần đây">{trace.recentTurnCount}</Stat>
          )}
          {trace.compactedTurnCount !== null && (
            <Stat label="Đã nén">{trace.compactedTurnCount}</Stat>
          )}
          {trace.backgroundStatus !== "idle" && (
            <Stat label="Tiến trình nền">
              <Badge variant="secondary">{trace.backgroundStatus}</Badge>
            </Stat>
          )}
        </div>
      )}

      {knowledge.memory?.summary !== undefined && knowledge.memory.summary !== "" && (
        <Field label="Tóm tắt phiên">
          <ScrollArea className="border-border h-28 rounded-lg border p-3">
            <p className="text-muted-foreground text-sm leading-6 whitespace-pre-wrap">
              {knowledge.memory.summary}
            </p>
          </ScrollArea>
        </Field>
      )}

      {/* The proposal inbox is rendered only when something is in it. Nothing in
          the production runtime creates proposals today, so an always-visible
          empty section was a control that could never do anything. */}
      {proposals.length > 0 && (
        <Field label={`Đề xuất chờ duyệt (${proposals.length})`}>
          <ul className="border-border divide-border flex flex-col divide-y rounded-lg border">
            {proposals.map((proposal) => (
              <li key={proposal.id} className="flex items-center gap-3 px-3 py-2.5 text-sm">
                <Badge variant="outline">{proposal.kind}</Badge>
                <span className="min-w-0 flex-1">{proposal.statement}</span>
                <span className="text-muted-foreground font-mono text-[10px]">
                  {proposal.confidence.toFixed(2)}
                </span>
                <Button size="sm" disabled={!connected} onClick={() => onApprove(proposal.id)}>
                  Duyệt
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  disabled={!connected}
                  onClick={() => onReject(proposal.id)}
                >
                  Bỏ
                </Button>
              </li>
            ))}
          </ul>
        </Field>
      )}

      {knowledge.lastAction !== null && !knowledge.lastAction.ok && (
        <p className="text-destructive text-sm">
          {knowledge.lastAction.action} thất bại · {knowledge.lastAction.errorCode ?? "không rõ"}
        </p>
      )}
    </Section>
  );
}

/**
 * The last retrieval.
 *
 * Kept because groundedness is the repository's top open blocker and a UI that
 * makes evidence cheap to check is the second line of defence. Nothing is
 * re-scored here; it shows what the engine decided.
 */
function RetrievalSection({ knowledge }: { knowledge: KnowledgeState }) {
  const trace = knowledge.retrieval;

  if (trace === null) {
    return (
      <Section icon={Search} title="Truy xuất gần nhất">
        <p className="text-muted-foreground text-sm leading-6">
          Chưa có lượt nào truy xuất. Một câu hỏi mà router xử lý thẳng thì không chạm tới vault —
          bảng trống ở đây là bình thường, không phải lỗi.
        </p>
      </Section>
    );
  }

  return (
    <Section
      icon={Search}
      title="Truy xuất gần nhất"
      description={trace.query !== "" ? trace.query : undefined}
      actions={
        <Badge variant="outline">
          {trace.tier} · {trace.latencyMs.toFixed(0)} ms
        </Badge>
      }
    >
      <p className="text-sm leading-6">{evidenceSummary(trace.evidence)}</p>

      {trace.evidence !== null && (
        <div className="flex flex-col gap-2">
          <Stat label="Điểm cao nhất">
            <span className="font-mono text-xs">{score(trace.evidence.top_score)}</span>
          </Stat>
          <Stat label="Khoảng cách với hạng 2">
            <span className="font-mono text-xs">{score(trace.evidence.margin)}</span>
          </Stat>
          <Stat label="Đoạn khớp">
            {trace.evidence.hit_count ?? 0} nhận · {trace.rejectedCount} loại
          </Stat>
        </div>
      )}

      {trace.columns.map((column) => (
        <Field key={column.source} label={column.source}>
          <ul className="border-border divide-border flex flex-col divide-y rounded-lg border">
            {column.hits.map((hit, index) => (
              <li
                key={`${hit.path}-${index}`}
                className="flex items-center gap-3 px-3 py-2 text-sm"
              >
                <span className="min-w-0 flex-1 truncate font-mono text-xs">{hit.path}</span>
                <span className="text-muted-foreground shrink-0 font-mono text-xs tabular-nums">
                  {score(hit.score)}
                </span>
              </li>
            ))}
          </ul>
        </Field>
      ))}
    </Section>
  );
}

export function KnowledgePanel(props: KnowledgePanelProps) {
  const { knowledge } = props;

  // Nothing at all yet: one clear next step instead of four empty sections.
  if (knowledge.vault === null && knowledge.indexJob === null) {
    return (
      <EmptyState
        icon={FolderOpen}
        title="Chưa có dữ liệu kiến thức"
        description="Engine chưa báo trạng thái vault. Trang này hiện thư mục tài liệu, chỉ mục truy xuất và bộ nhớ phiên."
        hint={props.connected ? "Đang chờ engine trả lời…" : "Engine chưa chạy."}
      />
    );
  }

  return (
    <div className="divide-border flex flex-col divide-y">
      <VaultSection {...props} />
      <IndexSection {...props} />
      <MemorySection {...props} />
      <RetrievalSection knowledge={knowledge} />
    </div>
  );
}
