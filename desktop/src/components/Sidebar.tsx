/**
 * Navigation.
 *
 * Replaces a 56 px icon rail. The rail could hold three unlabelled buttons, so
 * everything else in the app — retrieval, memory, session budget, voice
 * diagnostics, settings — was pushed into one sliding sheet subdivided by tabs.
 * That is what made it feel, in the user's words, jumbled: four unrelated
 * things behind one door.
 *
 * The reference app's answer, and now this one: those are **pages**, and the
 * sidebar is how you move between them. It also fixes a bug by construction —
 * voice used to cover the whole window, so during a call there was no way to
 * reach settings or the engine restart. A page cannot do that.
 *
 * Two things in the reference are deliberately not copied:
 *
 * * **A conversation list.** The protocol has no history command (`docs/18`
 *   §2), so a list of past chats would be invented data. What is real is the
 *   session running right now, and that is what the one entry shows.
 * * **An account footer.** There is no account. The footer carries the thing a
 *   local engine actually has: whether it is alive, and a way to restart it.
 */

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

import { Button } from "@/components/ui/button";
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
  /** Present when the session has turns; this is the only "history" that exists. */
  sessionTitle: string | null;
  connected: boolean;
  starting: boolean;
  voiceRunning: boolean;
  onRestartEngine: () => void;
  onCollapse: () => void;
}

export function Sidebar({
  page,
  onNavigate,
  onNewConversation,
  sessionTitle,
  connected,
  starting,
  voiceRunning,
  onRestartEngine,
  onCollapse,
}: SidebarProps) {
  return (
    <nav className="bg-sidebar border-sidebar-border flex w-64 shrink-0 flex-col border-r">
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
        <span className="text-sm font-medium">Sơn Ca</span>
      </div>

      <div className="px-3 pt-3">
        <Button
          variant="outline"
          className="h-10 w-full justify-start gap-2 rounded-lg font-normal"
          onClick={onNewConversation}
        >
          <Plus className="size-4" />
          Cuộc trò chuyện mới
        </Button>
      </div>

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

      <div className="border-sidebar-border mx-3 mt-4 border-t pt-4">
        {sessionTitle === null ? (
          <p className="text-muted-foreground px-3 text-xs leading-5">
            Chưa có lượt nào. Engine không lưu lịch sử giữa các phiên.
          </p>
        ) : (
          <button
            type="button"
            onClick={() => onNavigate("chat")}
            className={cn(
              "flex h-9 w-full items-center gap-3 rounded-lg px-3 text-sm transition-colors",
              page === "chat"
                ? "bg-sidebar-accent text-sidebar-accent-foreground"
                : "text-muted-foreground hover:bg-sidebar-accent/50 hover:text-foreground",
            )}
            title={sessionTitle}
          >
            <MessageSquare className="size-4 shrink-0" />
            <span className="truncate">{sessionTitle}</span>
          </button>
        )}
      </div>

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
    </nav>
  );
}
