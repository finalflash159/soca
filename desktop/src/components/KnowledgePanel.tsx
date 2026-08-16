/**
 * Phase 4 surface: retrieval inspector, memory, and vault/index management.
 *
 * The retrieval inspector is not a debug view. Groundedness is the repository's
 * top open blocker, and the plan's argument is that a UI which makes evidence
 * cheap to check is the second line of defence: people only verify when
 * verifying is one glance rather than one navigation. So the passages, their
 * per-backend scores and the gate's verdict sit on the same screen as the
 * answer's provenance — nothing is re-scored or re-judged here.
 */

import { ThinkingOrb } from "thinking-orbs";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PanelEmpty, PanelSection } from "@/components/PanelSection";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { KnowledgeState } from "@/engine/knowledge";
import { evidenceSummary, indexJobRunning, memoryModeSummary } from "@/engine/knowledge";

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

function RetrievalInspector({ knowledge }: { knowledge: KnowledgeState }) {
  const trace = knowledge.retrieval;

  return (
    <PanelSection
      title="Retrieval"
      description={trace?.query ?? "nothing retrieved yet"}
      status={
        trace !== null ? (
          <>
            <Badge variant="outline">tier {trace.tier}</Badge>
            <span className="text-muted-foreground text-xs">{trace.latencyMs.toFixed(0)} ms</span>
          </>
        ) : null
      }
    >
      <div className="flex flex-col gap-3">
        {trace === null ? (
          <PanelEmpty>
            A turn the router resolves to no capability never reaches the vault,
            so an empty inspector is a normal outcome rather than a fault.
          </PanelEmpty>
        ) : (
          <>
            <div className="bg-muted/40 rounded-md px-3 py-2 text-sm">
              {evidenceSummary(trace.evidence)}
              {trace.evidence !== null && (
                <div className="text-muted-foreground mt-1 flex flex-wrap gap-3 font-mono text-[10px]">
                  <span>status {trace.evidence.status ?? "—"}</span>
                  <span>top {score(trace.evidence.top_score)}</span>
                  <span>margin {score(trace.evidence.margin)}</span>
                  <span>hits {trace.evidence.hit_count ?? 0}</span>
                  <span>rejected {trace.rejectedCount}</span>
                  {trace.evidence.reason != null && <span>reason {String(trace.evidence.reason)}</span>}
                </div>
              )}
            </div>

            {trace.columns.map((column) => (
              <div key={column.source} className="flex flex-col gap-1">
                <span className="text-muted-foreground text-xs font-medium">{column.source}</span>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>path</TableHead>
                      <TableHead className="w-20 text-right">score</TableHead>
                      <TableHead className="w-20 text-right">sparse</TableHead>
                      <TableHead className="w-20 text-right">dense</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {column.hits.map((hit, index) => (
                      <TableRow key={`${hit.path}-${index}`}>
                        <TableCell className="font-mono text-[11px]">{hit.path}</TableCell>
                        <TableCell className="text-right font-mono text-[11px]">
                          {score(hit.score)}
                        </TableCell>
                        <TableCell className="text-right font-mono text-[11px]">
                          {score(hit.sparse_score)}
                        </TableCell>
                        <TableCell className="text-right font-mono text-[11px]">
                          {score(hit.dense_score)}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            ))}
          </>
        )}
      </div>
    </PanelSection>
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

  return (
    <PanelSection
      title="Memory"
      description={memoryModeSummary(trace)}
      status={
        trace !== null && trace.backgroundStatus !== "idle" ? (
          <Badge variant="secondary">{trace.backgroundStatus}</Badge>
        ) : null
      }
      action={
        <>
          <Button size="sm" variant="ghost" disabled={!connected} onClick={onRefreshMemory}>
            Refresh
          </Button>
          <Button size="sm" variant="outline" disabled={!connected} onClick={onCompact}>
            Compact
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-3">
        {trace !== null && (
          <div className="text-muted-foreground flex flex-wrap gap-3 font-mono text-[10px]">
            <span>worker {trace.summaryWorkerState}</span>
            {trace.recentTurnCount !== null && <span>recent {trace.recentTurnCount}</span>}
            {trace.compactedTurnCount !== null && <span>compacted {trace.compactedTurnCount}</span>}
            {trace.pendingCompaction && <span>compaction pending</span>}
          </div>
        )}

        {trace !== null && trace.hits.length > 0 && (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>id</TableHead>
                <TableHead className="w-24">corpus</TableHead>
                <TableHead className="w-20 text-right">relevance</TableHead>
                <TableHead className="w-20 text-right">recency</TableHead>
                <TableHead className="w-20 text-right">total</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {trace.hits.map((hit) => (
                <TableRow key={hit.id}>
                  <TableCell className="font-mono text-[11px]">{hit.id}</TableCell>
                  <TableCell className="text-[11px]">{hit.corpus}</TableCell>
                  <TableCell className="text-right font-mono text-[11px]">
                    {score(hit.relevance)}
                  </TableCell>
                  <TableCell className="text-right font-mono text-[11px]">
                    {score(hit.recency)}
                  </TableCell>
                  <TableCell className="text-right font-mono text-[11px]">
                    {score(hit.total)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}

        {knowledge.memory?.summary !== undefined && knowledge.memory.summary !== "" && (
          <ScrollArea className="h-24">
            <p className="text-muted-foreground text-xs whitespace-pre-wrap">
              {knowledge.memory.summary}
            </p>
          </ScrollArea>
        )}

        <div className="flex flex-col gap-2">
          <span className="text-muted-foreground text-xs font-medium">
            Proposals ({knowledge.proposals.length})
          </span>
          {knowledge.proposals.length === 0 ? (
            <PanelEmpty>
              Nothing in the production runtime creates memory proposals today,
              so an empty inbox is the expected state rather than a failure.
            </PanelEmpty>
          ) : (
            knowledge.proposals.map((proposal) => (
              <div
                key={proposal.id}
                className="flex items-center gap-2 rounded-md border px-3 py-2 text-sm"
              >
                <Badge variant="outline">{proposal.kind}</Badge>
                <span className="flex-1">{proposal.statement}</span>
                <span className="text-muted-foreground font-mono text-[10px]">
                  {proposal.confidence.toFixed(2)}
                </span>
                <Button size="sm" disabled={!connected} onClick={() => onApprove(proposal.id)}>
                  Approve
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  disabled={!connected}
                  onClick={() => onReject(proposal.id)}
                >
                  Reject
                </Button>
              </div>
            ))
          )}
          {knowledge.lastAction !== null && !knowledge.lastAction.ok && (
            <p className="text-destructive text-xs">
              {knowledge.lastAction.action} failed · {knowledge.lastAction.errorCode ?? "unknown"}
            </p>
          )}
        </div>
      </div>
    </PanelSection>
  );
}

function VaultSection({
  knowledge,
  connected,
  onInit,
  onIndex,
}: Pick<KnowledgePanelProps, "knowledge" | "connected" | "onInit" | "onIndex">) {
  const running = indexJobRunning(knowledge.indexJob);

  return (
    <PanelSection
      title="Vault & index"
      description={knowledge.vault ?? "no vault path yet"}
      /* The plan reserves `shaping` for index builds; this is that state. */
      status={running ? <ThinkingOrb state="shaping" size={20} /> : null}
      action={
        <>
          <Button size="sm" variant="outline" disabled={!connected || running} onClick={onInit}>
            Init
          </Button>
          <Button size="sm" disabled={!connected || running} onClick={onIndex}>
            Build index
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-2 text-sm">
        <p>
          Index{" "}
          {knowledge.indexPresent === null
            ? "unknown — ask for status"
            : knowledge.indexPresent
              ? "present"
              : "not built"}
        </p>
        {knowledge.indexJob !== null && (
          <p className="text-muted-foreground text-xs">
            {knowledge.indexJob.action} · {knowledge.indexJob.status} · {knowledge.indexJob.detail}
          </p>
        )}
        {knowledge.indexJob?.errorCode != null && (
          <p className="text-destructive text-xs">{knowledge.indexJob.errorCode}</p>
        )}
      </div>
    </PanelSection>
  );
}

export function KnowledgePanel(props: KnowledgePanelProps) {
  return (
    <div className="mx-auto flex w-full max-w-[46rem] flex-col gap-3">
      <RetrievalInspector knowledge={props.knowledge} />
      <MemorySection {...props} />
      <VaultSection {...props} />
    </div>
  );
}
