/**
 * The strip above every page.
 *
 * Carries the three things that are true no matter which page is open: how to
 * get the sidebar back, what this app is, and what the agent is doing right
 * now. The orb lives here for that last reason — plan §0.2 makes it the single
 * agent-state animation, and this is the one place on screen that never
 * changes, so it is the only place it can be *always* visible.
 */

import { Moon, PanelLeft, Sun } from "lucide-react";
import type { OrbState } from "thinking-orbs";
import { ThinkingOrb } from "thinking-orbs";

import { Button } from "@/components/ui/button";
import { orbLabel } from "@/engine/orb";

interface TopBarProps {
  orbState: OrbState;
  sidebarOpen: boolean;
  onOpenSidebar: () => void;
  onToggleTheme: () => void;
}

export function TopBar({ orbState, sidebarOpen, onOpenSidebar, onToggleTheme }: TopBarProps) {
  return (
    <header className="border-border flex h-14 shrink-0 items-center gap-3 border-b px-4">
      {!sidebarOpen && (
        <Button
          id="open-sidebar"
          size="sm"
          variant="ghost"
          className="text-muted-foreground hover:text-foreground size-8 rounded-lg p-0"
          title="Mở thanh bên"
          aria-label="Mở thanh bên"
          onClick={onOpenSidebar}
        >
          <PanelLeft className="size-4" />
        </Button>
      )}

      {/* Pinned to `solving` — this one is the mark beside the name, not a
          readout. `breathing` is a faint dotted ring that reads as an empty
          placeholder at 20 px, and the state it would report is already spelt
          out in words immediately to the right, so nothing is lost by making
          the picture constant and letting the label carry the truth. */}
      <div className="flex items-center gap-2.5">
        <ThinkingOrb state="solving" size={20} aria-hidden />
        <span className="text-[15px] font-medium">Sơn Ca</span>
      </div>

      <span className="text-muted-foreground ml-1 text-xs" role="status">
        {orbLabel(orbState)}
      </span>

      <div className="ml-auto flex items-center gap-1">
        <Button
          size="sm"
          variant="ghost"
          className="text-muted-foreground hover:text-foreground size-8 rounded-lg p-0"
          title="Đổi sáng/tối"
          aria-label="Đổi sáng/tối"
          onClick={onToggleTheme}
        >
          {/* Both render; CSS picks one, so the icon can never disagree with the
              theme that is actually applied. */}
          <Sun className="size-4 dark:hidden" />
          <Moon className="hidden size-4 dark:block" />
        </Button>
      </div>
    </header>
  );
}
