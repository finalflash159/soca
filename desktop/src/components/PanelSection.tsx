/**
 * The one section shell every panel uses.
 *
 * Open WebUI's v0.11.0 note is the argument (`plan` §5.6.1): their sidebar has
 * six groups built on a single section component, so every header, chevron and
 * hover state behaves identically and the user learns one behaviour instead of
 * six. SoCa has Knowledge, Memory, Vault, Providers, Profiles and Voice — the
 * same risk, so the same answer.
 *
 * `action` sits in the header rather than floating in the body, and only the
 * sections that actually create or run something get one — §5.6.2, no "+"
 * scattered everywhere.
 */

import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

interface PanelSectionProps {
  title: string;
  /** One short line. Long explanations belong in docs, not in a header. */
  description?: string;
  action?: ReactNode;
  /** Status chip shown before the action. */
  status?: ReactNode;
  children: ReactNode;
  className?: string;
}

export function PanelSection({
  title,
  description,
  action,
  status,
  children,
  className,
}: PanelSectionProps) {
  return (
    <section
      className={cn("border-border/60 bg-card/40 rounded-lg border", className)}
    >
      <header className="flex items-center gap-3 px-4 py-3">
        <div className="flex min-w-0 flex-col">
          <h2 className="text-sm font-medium tracking-tight">{title}</h2>
          {description !== undefined && (
            <p className="text-muted-foreground truncate text-xs">{description}</p>
          )}
        </div>
        <div className="ml-auto flex shrink-0 items-center gap-2">
          {status}
          {action}
        </div>
      </header>
      <div className="border-border/60 border-t px-4 py-3">{children}</div>
    </section>
  );
}

/** Empty state with the same voice everywhere: say what is true, not "oops". */
export function PanelEmpty({ children }: { children: ReactNode }) {
  return <p className="text-muted-foreground py-2 text-xs leading-relaxed">{children}</p>;
}

/** Label/value row, so every panel reads the same way. */
export function PanelRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline gap-3 py-1 text-xs">
      <span className="text-muted-foreground w-28 shrink-0">{label}</span>
      <span className="min-w-0 flex-1">{children}</span>
    </div>
  );
}
