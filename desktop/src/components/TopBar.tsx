/**
 * The strip above every page.
 *
 * Carries the three things that are true no matter which page is open: how to
 * get the sidebar back, what this app is, and what the agent is doing right
 * now. The orb lives here for that last reason — plan §0.2 makes it the single
 * agent state. The state is announced in text, while the official parrot mark
 * remains a stable anchor rather than pretending to be a progress indicator.
 */

import { Moon, PanelLeft, Sun } from "lucide-react";

import { ActivityOrb } from "@/components/ActivityOrb";
import { BrandMark } from "@/components/BrandMark";
import { Button } from "@/components/ui/button";
import { orbLabel, type OrbState } from "@/engine/orb";

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

      <BrandMark className="text-[15px] font-medium" iconClassName="text-base" />

      <div className="text-muted-foreground ml-1 flex items-center gap-1.5 text-xs" role="status" aria-atomic="true">
        <ActivityOrb state={orbState} size={18} />
        <span>{orbLabel(orbState)}</span>
      </div>

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
