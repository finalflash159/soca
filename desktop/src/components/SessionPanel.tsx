/** Context and usage disclosures for the active conversation. */

import { PanelEmpty, PanelRow, PanelSection } from "@/components/PanelSection";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import type { SessionState } from "@/engine/session";
import { budgetUsedFraction, contextBasis } from "@/engine/session";

interface SessionPanelProps {
  session: SessionState;
  connected: boolean;
  onRefresh: () => void;
}

function tokens(value: number | null): string {
  return value === null ? "—" : value.toLocaleString("vi-VN");
}

export function SessionPanel({ session, connected, onRefresh }: SessionPanelProps) {
  const { context, usage } = session;
  const used = budgetUsedFraction(context);

  return (
    <div className="flex w-full flex-col gap-3">
      <PanelSection
        title="Ngân sách ngữ cảnh"
        description={contextBasis(context)}
        status={
          context !== null && context.estimated ? <Badge variant="outline">ước lượng</Badge> : null
        }
        action={
          <Button size="sm" variant="ghost" disabled={!connected} onClick={onRefresh}>
            Cập nhật
          </Button>
        }
      >
        {context === null ? (
          <PanelEmpty>Chưa có số liệu. Bấm Cập nhật hoặc gõ /context.</PanelEmpty>
        ) : !context.ready ? (
          <p className="text-destructive text-xs">
            {context.error ?? "không dựng được manifest"}
            {context.errorDetail !== null && ` · ${context.errorDetail}`}
          </p>
        ) : (
          <>
            {used !== null && (
              <div className="mb-3 flex flex-col gap-1">
                <Progress value={used * 100} />
                <span className="text-muted-foreground text-[10px]">
                  {tokens(context.residentPromptTokens)} / {tokens(context.inputBudgetTokens)} token
                  đầu vào đã dùng
                </span>
              </div>
            )}
            <PanelRow label="Cửa sổ ngữ cảnh">{tokens(context.modelContextTokens)}</PanelRow>
            <PanelRow label="Dành cho phản hồi">{tokens(context.outputReserveTokens)}</PanelRow>
            <PanelRow label="Còn cho nội dung mới">{tokens(context.availableDynamicTokens)}</PanelRow>
            <PanelRow label="Provider báo cáo">
              {/* Null until a provider reports real counts — §4 forbids showing
                  the estimate in this slot. */}
              {context.providerPromptTokens === null ? (
                <span className="text-muted-foreground">chưa có</span>
              ) : (
                tokens(context.providerPromptTokens)
              )}
            </PanelRow>
          </>
        )}
      </PanelSection>

      {context?.ready === true && context.components.length > 0 && (
        <PanelSection title="Thành phần prompt" description={`${context.components.length} khối`}>
          <div className="flex flex-col gap-1">
            {context.components.map((component, index) => (
              <div key={index} className="flex items-baseline gap-3 text-xs">
                <span className="flex-1 truncate">
                  {String(component.name ?? `khối ${index + 1}`)}
                </span>
                <span className="text-muted-foreground font-mono">
                  {tokens(typeof component.tokens === "number" ? component.tokens : null)}
                </span>
              </div>
            ))}
          </div>
        </PanelSection>
      )}

      <PanelSection
        title="Mức dùng"
        description="Cộng dồn từ lúc engine khởi động"
        action={
          <Button size="sm" variant="ghost" disabled={!connected} onClick={onRefresh}>
            Cập nhật
          </Button>
        }
      >
        {usage === null ? (
          <PanelEmpty>Chưa có số liệu. Bấm Cập nhật hoặc gõ /usage.</PanelEmpty>
        ) : (
          <>
            <PanelRow label="Lượt">
              {usage.turns} ({usage.llmTurns} gọi LLM)
            </PanelRow>
            <PanelRow label="Token đầu vào">{tokens(usage.promptTokens)}</PanelRow>
            <PanelRow label="Token phản hồi">{tokens(usage.completionTokens)}</PanelRow>
            <PanelRow label="TTFT trung bình">
              {usage.meanTtftMs === null ? "—" : `${Math.round(usage.meanTtftMs)} ms`}
            </PanelRow>
            <PanelRow label="Token/giây">
              {usage.meanTokensPerSecond === null ? "—" : usage.meanTokensPerSecond.toFixed(1)}
            </PanelRow>
          </>
        )}
      </PanelSection>
    </div>
  );
}
