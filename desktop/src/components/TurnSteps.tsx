/**
 * What the turn did, before it answered.
 *
 * The reference app's chat screen puts a collapsed pill above the answer — an
 * icon, a count, a chevron — that opens into a dotted timeline of the steps
 * taken. Filled dots are work that completed, hollow is the step still running.
 *
 * Every row here is a `turn_progress.phase` the engine actually emitted. There
 * is no per-step timing or detail in the protocol, so none is shown: a duration
 * next to each row would be invented, and inventing plausible numbers on an
 * evidence panel is worse than having no panel.
 *
 * Collapsed by default. On an ordinary chat turn the trail is `preparing →
 * analyzing → synthesis`, which is noise; it earns its space on the retrieval
 * and tool turns, where nothing else explains a ten-second wait.
 */

import { ChevronDown, ListChecks } from "lucide-react";
import { useState } from "react";

import { phaseLabel } from "@/engine/conversation";
import { cn } from "@/lib/utils";

export function TurnSteps({ steps, running }: { steps: string[]; running: boolean }) {
  const [open, setOpen] = useState(false);

  if (steps.length === 0) {
    return null;
  }

  return (
    <div className="flex flex-col gap-2">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        className="border-border bg-card text-muted-foreground hover:text-foreground flex w-fit items-center gap-2 rounded-lg border px-2.5 py-1.5 text-xs transition-colors"
      >
        <ListChecks className="size-3.5" />
        <span className="tabular-nums">{steps.length}</span>
        <ChevronDown className={cn("size-3.5 transition-transform", open && "rotate-180")} />
      </button>

      {open && (
        <ol className="border-border ml-[7px] flex flex-col gap-2.5 border-l pl-4">
          {steps.map((step, index) => {
            // Only the last step can still be in flight, and only while the
            // turn is open.
            const active = running && index === steps.length - 1;
            return (
              <li key={`${step}-${index}`} className="relative text-sm">
                <span
                  className={cn(
                    "absolute top-[7px] -left-[21px] size-2 rounded-full",
                    active ? "bg-primary animate-pulse" : "bg-chart-3",
                  )}
                  aria-hidden
                />
                <span className={active ? "text-foreground" : "text-muted-foreground"}>
                  {phaseLabel(step)}
                </span>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
