/** Primary navigation and the engine-owned saved-session workspace. */

import {
  AudioLines,
  BookOpen,
  Gauge,
  MessageSquare,
  PanelLeft,
  Plus,
  RotateCcw,
  Settings,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { BrandMark } from "@/components/BrandMark";
import { Button } from "@/components/ui/button";
import { SessionList } from "@/components/SessionList";
import type { SessionHistoryState, SessionSummary } from "@/engine/session-history";
import { cn } from "@/lib/utils";

export type PageId = "chat" | "voice" | "knowledge" | "session" | "settings";

interface NavEntry {
  id: PageId;
  label: string;
  icon: LucideIcon;
}

const NAV: NavEntry[] = [
  { id: "chat", label: "Trò chuyện", icon: MessageSquare },
  { id: "voice", label: "Thoại", icon: AudioLines },
  { id: "knowledge", label: "Kiến thức", icon: BookOpen },
  { id: "session", label: "Phiên", icon: Gauge },
  { id: "settings", label: "Cài đặt", icon: Settings },
];

interface SidebarProps {
  page: PageId;
  onNavigate: (page: PageId) => void;
  onNewConversation: () => void;
  sessions: SessionHistoryState;
  connected: boolean;
  starting: boolean;
  voiceRunning: boolean;
  sessionBusy: boolean;
  newConversationDisabled: boolean;
  onRestartEngine: () => void;
  onCollapse: () => void;
  onOpenSession: (session: SessionSummary) => void;
  onRenameSession: (session: SessionSummary, title: string) => void;
  onDeleteSession: (session: SessionSummary) => void;
  onLoadMoreSessions: () => void;
  onOpenSessionSettings: () => void;
}

export function Sidebar({
  page,
  onNavigate,
  onNewConversation,
  sessions,
  connected,
  starting,
  voiceRunning,
  sessionBusy,
  newConversationDisabled,
  onRestartEngine,
  onCollapse,
  onOpenSession,
  onRenameSession,
  onDeleteSession,
  onLoadMoreSessions,
  onOpenSessionSettings,
}: SidebarProps) {
  const rootRef = useRef<HTMLElement>(null);
  const [compact, setCompact] = useState(() => window.matchMedia("(max-width: 760px)").matches);

  useEffect(() => {
    const media = window.matchMedia("(max-width: 760px)");
    const update = () => setCompact(media.matches);
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);

  useEffect(() => {
    if (!compact) return;
    const root = rootRef.current;
    if (root === null) return;
    const focusable = () => Array.from(
      root.querySelectorAll<HTMLElement>(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      ),
    );
    const frame = window.requestAnimationFrame(() => focusable()[0]?.focus());
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onCollapse();
        return;
      }
      if (event.key !== "Tab") return;
      const controls = focusable();
      const first = controls[0];
      const last = controls[controls.length - 1];
      if (first === undefined || last === undefined) return;
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    root.addEventListener("keydown", onKeyDown);
    return () => {
      window.cancelAnimationFrame(frame);
      root.removeEventListener("keydown", onKeyDown);
    };
  }, [compact, onCollapse]);

  return (
    <aside
      ref={rootRef}
      role={compact ? "dialog" : undefined}
      aria-modal={compact || undefined}
      aria-label={compact ? "Thanh điều hướng" : undefined}
      className="bg-sidebar border-sidebar-border flex w-64 shrink-0 flex-col border-r max-[760px]:fixed max-[760px]:inset-y-0 max-[760px]:left-0 max-[760px]:z-50 max-[760px]:shadow-xl"
    >
      <div className="flex items-center gap-1 px-3 pt-3">
        <Button
          size="sm"
          variant="ghost"
          className="text-muted-foreground hover:text-foreground size-8 shrink-0 rounded-lg p-0"
          title="Thu gọn thanh bên"
          aria-label="Thu gọn thanh bên"
          onClick={onCollapse}
        >
          <PanelLeft className="size-4" />
        </Button>
        <BrandMark className="text-sm font-medium" iconClassName="text-base" />
      </div>

      <div className="px-3 pt-3">
        <Button
          id="new-conversation"
          variant="outline"
          className="h-10 w-full justify-start gap-2 rounded-lg font-normal"
          disabled={newConversationDisabled}
          onClick={onNewConversation}
        >
          <Plus className="size-4" />
          Cuộc trò chuyện mới
        </Button>
      </div>

      <nav aria-label="Điều hướng chính">
        <ul className="flex flex-col gap-0.5 px-3 pt-4">
          {NAV.map((entry) => {
            const active = page === entry.id;
            return (
              <li key={entry.id}>
                <button
                  type="button"
                  aria-current={active ? "page" : undefined}
                  onClick={() => onNavigate(entry.id)}
                  className={cn(
                    "flex h-9 w-full items-center gap-3 rounded-lg px-3 text-sm transition-colors",
                    active
                      ? "bg-sidebar-accent text-sidebar-accent-foreground font-medium"
                      : "text-muted-foreground hover:bg-sidebar-accent/50 hover:text-foreground",
                  )}
                >
                  <entry.icon className="size-[18px] shrink-0" />
                  <span className="truncate">{entry.label}</span>
                  {entry.id === "voice" && voiceRunning && (
                    <span
                      className="bg-primary ml-auto size-1.5 shrink-0 rounded-full"
                      title="Mic đang mở"
                      aria-label="Mic đang mở"
                    />
                  )}
                </button>
              </li>
            );
          })}
        </ul>
      </nav>

      <SessionList
        history={sessions}
        disabled={!connected || sessionBusy}
        onOpen={onOpenSession}
        onRename={onRenameSession}
        onDelete={onDeleteSession}
        onLoadMore={onLoadMoreSessions}
        onOpenSettings={onOpenSessionSettings}
      />

      {/* Engine health, where the reference app puts the signed-in user. There
          is no account here; what a local engine has is a pulse. */}
      <div className="border-sidebar-border mt-auto flex items-center gap-2.5 border-t px-4 py-3">
        <span
          className={cn(
            "size-2 shrink-0 rounded-full",
            connected ? "bg-chart-3" : starting ? "bg-chart-4" : "bg-muted-foreground/40",
          )}
          aria-hidden
        />
        <span className="text-muted-foreground min-w-0 flex-1 truncate text-xs">
          {connected ? "Engine đang chạy" : starting ? "Đang khởi động…" : "Engine đã dừng"}
        </span>
        <Button
          size="sm"
          variant="ghost"
          className="text-muted-foreground hover:text-foreground size-7 shrink-0 rounded-md p-0"
          title="Khởi động lại engine"
          aria-label="Khởi động lại engine"
          onClick={onRestartEngine}
        >
          <RotateCcw className="size-3.5" />
        </Button>
      </div>
    </aside>
  );
}
