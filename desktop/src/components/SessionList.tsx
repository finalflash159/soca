import { Check, CircleEllipsis, History, LoaderCircle, Pencil, Trash2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import type { SessionHistoryState, SessionSummary } from "@/engine/session-history";
import { cn } from "@/lib/utils";

interface SessionListProps {
  history: SessionHistoryState;
  disabled: boolean;
  onOpen: (session: SessionSummary) => void;
  onRename: (session: SessionSummary, title: string) => void;
  onDelete: (session: SessionSummary) => void;
  onLoadMore: () => void;
  onOpenSettings: () => void;
}

function relativeTime(value: string): string {
  const time = Date.parse(value);
  if (!Number.isFinite(time)) {
    return "đã lưu";
  }
  const minutes = Math.max(0, Math.round((Date.now() - time) / 60_000));
  if (minutes < 1) return "vừa xong";
  if (minutes < 60) return `${minutes} phút trước`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} giờ trước`;
  return `${Math.round(hours / 24)} ngày trước`;
}

function SessionRow({
  session,
  active,
  disabled,
  operation,
  onOpen,
  onRename,
  onDelete,
}: {
  session: SessionSummary;
  active: boolean;
  disabled: boolean;
  operation: SessionHistoryState["operation"];
  onOpen: () => void;
  onRename: (title: string) => void;
  onDelete: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [title, setTitle] = useState(session.title);
  const inputRef = useRef<HTMLInputElement>(null);
  const renamePending =
    operation?.action === "rename" && operation.sessionId === session.sessionId && operation.status === "started";

  useEffect(() => {
    if (editing) {
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [editing]);

  useEffect(() => {
    if (
      editing &&
      operation?.action === "rename" &&
      operation.sessionId === session.sessionId &&
      operation.status === "completed"
    ) {
      setEditing(false);
    }
  }, [editing, operation, session.sessionId]);

  if (editing) {
    return (
      <li className="px-1">
        <form
          className="bg-accent flex items-center gap-1 rounded-lg p-1"
          onSubmit={(event) => {
            event.preventDefault();
            const next = title.trim();
            if (next !== "" && next !== session.title) onRename(next);
            if (next === session.title) setEditing(false);
          }}
        >
          <input
            ref={inputRef}
            value={title}
            maxLength={48}
            aria-label={`Đổi tên phiên ${session.title}`}
            disabled={disabled || renamePending}
            onChange={(event) => setTitle(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Escape") {
                event.preventDefault();
                setTitle(session.title);
                setEditing(false);
              }
            }}
            className="border-input bg-background h-8 min-w-0 flex-1 rounded-md border px-2 text-xs outline-none focus-visible:ring-2 focus-visible:ring-ring/60"
          />
          <Button
            type="submit"
            size="icon-xs"
            variant="ghost"
            disabled={disabled || renamePending || title.trim() === ""}
            aria-label="Lưu tên phiên"
          >
            {renamePending ? <LoaderCircle className="animate-spin" /> : <Check />}
          </Button>
          <Button
            type="button"
            size="icon-xs"
            variant="ghost"
            disabled={renamePending}
            aria-label="Hủy đổi tên"
            onClick={() => {
              setTitle(session.title);
              setEditing(false);
            }}
          >
            ×
          </Button>
        </form>
      </li>
    );
  }

  return (
    <li className="group/session px-1">
      <div
        className={cn(
          "flex items-center gap-1 rounded-lg",
          active && "bg-accent text-accent-foreground",
        )}
      >
        <button
          id={`session-select-${session.sessionId}`}
          type="button"
          aria-current={active ? "page" : undefined}
          disabled={disabled}
          onClick={onOpen}
          className={cn(
            "flex h-11 min-w-0 flex-1 flex-col justify-center rounded-lg px-2 text-left transition-colors",
            active
              ? "cursor-default focus-visible:ring-2 focus-visible:ring-ring/70"
              : "hover:bg-accent/70 focus-visible:ring-2 focus-visible:ring-ring/70",
            "disabled:opacity-100",
          )}
        >
          <span className="flex min-w-0 items-center gap-1.5 text-xs font-medium">
            {active && <span className="bg-primary size-1.5 shrink-0 rounded-full" aria-hidden />}
            <span className="truncate">{session.title}</span>
          </span>
          <span className="text-muted-foreground truncate text-[10px]">
            {session.checkpointOnly ? "Chỉ khôi phục context" : relativeTime(session.updatedAt)}
          </span>
        </button>
        <DropdownMenu>
          <DropdownMenuTrigger
            render={
              <Button
                size="icon-xs"
                variant="ghost"
                className="text-muted-foreground mr-1 opacity-0 transition-opacity group-hover/session:opacity-100 focus-visible:opacity-100"
                aria-label={`Thao tác cho phiên ${session.title}`}
                disabled={disabled}
              />
            }
          >
            <CircleEllipsis />
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-36">
            <DropdownMenuItem onClick={() => setEditing(true)}>
              <Pencil /> Đổi tên
            </DropdownMenuItem>
            <DropdownMenuItem variant="destructive" onClick={onDelete}>
              <Trash2 /> Xóa vĩnh viễn
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </li>
  );
}

export function SessionList({
  history,
  disabled,
  onOpen,
  onRename,
  onDelete,
  onLoadMore,
  onOpenSettings,
}: SessionListProps) {
  const [deleteTarget, setDeleteTarget] = useState<SessionSummary | null>(null);
  const deleteOpener = useRef<HTMLElement | null>(null);
  const deletedSessionId = useRef<string | null>(null);
  const operationPending = history.operation?.status === "started";

  const closeDelete = () => {
    const opener = deleteOpener.current;
    setDeleteTarget(null);
    requestAnimationFrame(() => opener?.focus());
  };

  useEffect(() => {
    if (
      deletedSessionId.current === null ||
      history.operation?.action !== "delete" ||
      history.operation.status === "started"
    ) {
      return;
    }
    const deleted = deletedSessionId.current;
    deletedSessionId.current = null;
    if (history.operation.status !== "completed") return;
    requestAnimationFrame(() => {
      const next = history.activeSessionId;
      const target =
        next === null ? null : document.getElementById(`session-select-${next}`);
      (target ?? document.getElementById("new-conversation"))?.focus();
    });
    // Keep the deleted identity in scope: it documents that focus must never
    // return to a removed destructive action.
    void deleted;
  }, [history.activeSessionId, history.operation]);

  const empty =
    history.persistence === "ram_only" ? (
      <div className="px-3 py-2.5">
        <p className="text-muted-foreground text-xs leading-5">
          Phiên này chỉ tồn tại đến khi đóng ứng dụng.
        </p>
        <button
          type="button"
          className="text-primary mt-1 text-xs font-medium underline-offset-4 hover:underline focus-visible:rounded-sm focus-visible:outline-2 focus-visible:outline-ring"
          onClick={onOpenSettings}
        >
          Bật lưu phiên trên máy
        </button>
      </div>
    ) : (
      <p className="text-muted-foreground px-3 py-2.5 text-xs leading-5">Chưa có phiên đã lưu.</p>
    );

  return (
    <section className="border-border mt-2 border-t pt-3" aria-labelledby="saved-sessions">
      <div className="flex items-center justify-between px-1 pb-1.5">
        <h2 id="saved-sessions" className="text-muted-foreground text-[11px] font-medium tracking-wide uppercase">
          Phiên đã lưu
        </h2>
        {history.listState === "loading" && <LoaderCircle className="text-muted-foreground size-3 animate-spin" aria-label="Đang tải phiên" />}
      </div>

      {history.listState === "error" ? (
        <div className="border-destructive/40 bg-destructive/5 rounded-lg border px-2.5 py-2 text-xs" role="alert">
          <p className="text-destructive leading-5">{history.listError ?? "Không thể tải phiên đã lưu."}</p>
          <button
            type="button"
            className="text-foreground mt-1 font-medium underline-offset-4 hover:underline"
            onClick={onLoadMore}
          >
            Thử lại
          </button>
        </div>
      ) : history.listState === "loading" && history.sessions.length === 0 ? (
        <div className="flex flex-col gap-1 px-1 py-1" aria-busy="true" aria-label="Đang tải danh sách phiên">
          <div className="bg-muted h-10 animate-pulse rounded-lg" />
          <div className="bg-muted h-10 animate-pulse rounded-lg" />
        </div>
      ) : history.sessions.length === 0 ? (
        empty
      ) : (
        <>
          <ul className="soca-scrollbar flex max-h-64 flex-col gap-0.5 overflow-y-auto" aria-label="Danh sách phiên đã lưu">
            {history.sessions.map((session) => (
              <SessionRow
                key={session.sessionId}
                session={session}
                active={history.activeSessionId === session.sessionId}
                disabled={disabled || operationPending}
                operation={history.operation}
                onOpen={() => onOpen(session)}
                onRename={(title) => onRename(session, title)}
                onDelete={() => {
                  deleteOpener.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
                  setDeleteTarget(session);
                }}
              />
            ))}
          </ul>
          {history.nextCursor !== null && (
            <Button
              size="sm"
              variant="ghost"
              className="text-muted-foreground mt-1 w-full"
              disabled={disabled || history.listState === "loading"}
              onClick={onLoadMore}
            >
              <History /> Tải thêm
            </Button>
          )}
        </>
      )}

      <Dialog
        open={deleteTarget !== null}
        onOpenChange={(open) => {
          if (!open) closeDelete();
        }}
      >
        <DialogContent showCloseButton={false}>
          <DialogHeader>
            <DialogTitle>Xóa vĩnh viễn phiên?</DialogTitle>
            <DialogDescription>
              “{deleteTarget?.title}” và {deleteTarget?.turnCount ?? 0} lượt trò chuyện sẽ bị xóa khỏi máy này.
              Thao tác này không thể hoàn tác.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <DialogClose render={<Button variant="outline" />} disabled={operationPending}>
              Hủy
            </DialogClose>
            <Button
              variant="destructive"
              disabled={deleteTarget === null || disabled || operationPending}
              onClick={() => {
                if (deleteTarget !== null) {
                  deletedSessionId.current = deleteTarget.sessionId;
                  onDelete(deleteTarget);
                  setDeleteTarget(null);
                }
              }}
            >
              Xóa vĩnh viễn
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </section>
  );
}
