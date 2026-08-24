/** Stable page chrome: navigation, brand, and theme. */

import { Moon, PanelLeft, Sun } from "lucide-react";

import { BrandMark } from "@/components/BrandMark";
import { Button } from "@/components/ui/button";

interface TopBarProps {
  sidebarOpen: boolean;
  onOpenSidebar: () => void;
  onToggleTheme: () => void;
}

export function TopBar({ sidebarOpen, onOpenSidebar, onToggleTheme }: TopBarProps) {
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
