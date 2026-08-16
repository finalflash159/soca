/**
 * Prompt budget and session usage.
 *
 * The engine has always answered `context` and `usage`; nothing rendered them,
 * so `/context` and `/usage` were commands that appeared to do nothing.
 *
 * The one rule that shapes this panel comes from `docs/18-engine-protocol.md`
 * §4: an estimated manifest must never be presented as observed usage. So the
 * basis is stated in the header rather than hidden, and the observed figures
 * stay blank until a provider reports them instead of falling back to the
 * estimate.
 */

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
        title="Prompt budget"
        description={contextBasis(context)}
        status={
          context !== null && context.estimated ? <Badge variant="outline">ước lượng</Badge> : null
        }
        action={
          <Button size="sm" variant="ghost" disabled={!connected} onClick={onRefresh}>
            Refresh
          </Button>
        }
      >
        {context === null ? (
          <PanelEmpty>Chưa nạp. Bấm Refresh hoặc gõ /context.</PanelEmpty>
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
            <PanelRow label="context window">{tokens(context.modelContextTokens)}</PanelRow>
            <PanelRow label="dành cho output">{tokens(context.outputReserveTokens)}</PanelRow>
            <PanelRow label="còn cho động">{tokens(context.availableDynamicTokens)}</PanelRow>
            <PanelRow label="provider đếm">
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
        title="Usage"
        description="Cộng dồn từ lúc engine khởi động"
        action={
          <Button size="sm" variant="ghost" disabled={!connected} onClick={onRefresh}>
            Refresh
          </Button>
        }
      >
        {usage === null ? (
          <PanelEmpty>Chưa nạp. Bấm Refresh hoặc gõ /usage.</PanelEmpty>
        ) : (
          <>
            <PanelRow label="lượt">
              {usage.turns} ({usage.llmTurns} gọi LLM)
            </PanelRow>
            <PanelRow label="prompt token">{tokens(usage.promptTokens)}</PanelRow>
            <PanelRow label="completion token">{tokens(usage.completionTokens)}</PanelRow>
            <PanelRow label="TTFT trung bình">
              {usage.meanTtftMs === null ? "—" : `${Math.round(usage.meanTtftMs)} ms`}
            </PanelRow>
            <PanelRow label="token/giây">
              {usage.meanTokensPerSecond === null ? "—" : usage.meanTokensPerSecond.toFixed(1)}
            </PanelRow>
          </>
        )}
      </PanelSection>
    </div>
  );
}
