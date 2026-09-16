import { ChevronDown, Clock3, Gauge } from "lucide-react";
import { useState } from "react";

import { SessionList } from "@/components/SessionList";
import { SessionPanel } from "@/components/SessionPanel";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { SessionState } from "@/engine/session";
import type {
  SessionHistoryState,
  SessionSummary,
} from "@/engine/session-history";
import { budgetUsedFraction } from "@/engine/session";
import { cn } from "@/lib/utils";

interface SessionContextProps {
  session: SessionState;
  history: SessionHistoryState;
  connected: boolean;
  busy: boolean;
  onRefresh: () => void;
  onOpenSession: (session: SessionSummary) => void;
  onRenameSession: (session: SessionSummary, title: string) => void;
  onDeleteSession: (session: SessionSummary) => void;
  onLoadMoreSessions: () => void;
  onOpenSessionSettings: () => void;
}

function usageLabel(session: SessionState): string {
  const used = budgetUsedFraction(session.context);
  if (used !== null) return `${Math.round(used * 100)}% ngữ cảnh`;
  if (session.usage !== null) return `${session.usage.turns} lượt`;
  return "Chưa có số liệu";
}

/**
 * Context belongs to the conversation surface, not the global navigation.
 * The disclosure is deliberately compact when closed: it answers “how much
 * context is this conversation using?” without pretending to be a dashboard.
 */
export function SessionContext({
  session,
  history,
  connected,
  busy,
  onRefresh,
  onOpenSession,
  onRenameSession,
  onDeleteSession,
  onLoadMoreSessions,
  onOpenSessionSettings,
}: SessionContextProps) {
  const [open, setOpen] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);

  return (
    <>
      <Collapsible
        open={open}
        onOpenChange={(next) => {
          setOpen(next);
          if (next) onRefresh();
        }}
        className="mx-auto w-full max-w-2xl"
      >
        <div className="flex items-center gap-1">
          <CollapsibleTrigger
            render={
              <Button
                size="sm"
                variant="ghost"
                className="text-muted-foreground hover:text-foreground h-8 gap-1.5 rounded-md px-2 text-xs"
              />
            }
          >
            <Gauge className="size-3.5" aria-hidden="true" />
            <span>Ngữ cảnh</span>
            <span className="text-muted-foreground/80 tabular-nums">
              {usageLabel(session)}
            </span>
            <ChevronDown
              className={cn(
                "size-3.5 transition-transform",
                open && "rotate-180",
              )}
              aria-hidden="true"
            />
          </CollapsibleTrigger>
          <Button
            size="sm"
            variant="ghost"
            className="text-muted-foreground hover:text-foreground h-8 gap-1.5 rounded-md px-2 text-xs"
            onClick={() => setHistoryOpen(true)}
          >
            <Clock3 className="size-3.5" aria-hidden="true" />
            Phiên đã lưu
          </Button>
        </div>
        <CollapsibleContent className="overflow-hidden data-open:animate-in data-open:fade-in-0 data-open:slide-in-from-top-1 data-closed:animate-out data-closed:fade-out-0 data-closed:slide-out-to-top-1">
          <div className="border-border bg-card mt-1 rounded-xl border p-3">
            <SessionPanel
              session={session}
              connected={connected}
              onRefresh={onRefresh}
            />
          </div>
        </CollapsibleContent>
      </Collapsible>

      <Dialog open={historyOpen} onOpenChange={setHistoryOpen}>
        <DialogContent className="max-h-[min(42rem,calc(100vh-2rem))] sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Phiên đã lưu</DialogTitle>
            <DialogDescription>
              Chọn một cuộc trò chuyện để tiếp tục. Audio và ASR partial không
              được lưu.
            </DialogDescription>
          </DialogHeader>
          <div className="min-h-0 max-h-[28rem] overflow-y-auto">
            <SessionList
              history={history}
              disabled={!connected || busy}
              onOpen={(item) => {
                setHistoryOpen(false);
                onOpenSession(item);
              }}
              onRename={onRenameSession}
              onDelete={(item) => {
                setHistoryOpen(false);
                onDeleteSession(item);
              }}
              onLoadMore={onLoadMoreSessions}
              onOpenSettings={() => {
                setHistoryOpen(false);
                onOpenSessionSettings();
              }}
            />
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
