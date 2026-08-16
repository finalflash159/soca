/**
 * The parts every page is built from.
 *
 * Taken from the reference app's page anatomy, which is the same on all four
 * of its screens: a title with a count beside it, actions on the right, then
 * either content or a centred empty state. Having them as components rather
 * than as a convention is what stops the fifth page from inventing a fifth
 * heading size.
 */

import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

/**
 * Page title row.
 *
 * The count sits in a chip next to the title rather than in parentheses after
 * it, so a page with nothing in it still reads as a page rather than as a
 * failed load. `count` is deliberately allowed to be `0`.
 */
export function PageHeader({
  title,
  count,
  description,
  actions,
}: {
  title: string;
  count?: number;
  description?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="flex items-start gap-4 pb-6">
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2.5">
          <h1 className="truncate text-2xl font-semibold tracking-tight">{title}</h1>
          {count !== undefined && (
            <span className="bg-secondary text-muted-foreground rounded-md px-2 py-0.5 text-sm tabular-nums">
              {count}
            </span>
          )}
        </div>
        {description !== undefined && (
          <p className="text-muted-foreground mt-1.5 text-sm">{description}</p>
        )}
      </div>
      {actions !== undefined && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}

/**
 * Centred empty state.
 *
 * Three lines, in this order and no other: what is not here, what the thing
 * would do, and how to get one. The old one-line grey `PanelEmpty` answered
 * only the first, which is the least useful of the three.
 */
export function EmptyState({
  icon: Icon,
  title,
  description,
  hint,
  action,
}: {
  icon: LucideIcon;
  title: string;
  description: string;
  hint?: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-4 px-8 py-16 text-center">
      <div className="border-border text-muted-foreground flex size-14 items-center justify-center rounded-full border">
        <Icon className="size-6" strokeWidth={1.5} />
      </div>
      <div className="flex max-w-md flex-col gap-2">
        <h2 className="text-xl font-medium">{title}</h2>
        <p className="text-muted-foreground text-sm leading-6">{description}</p>
        {hint !== undefined && <p className="text-muted-foreground/70 text-sm leading-6">{hint}</p>}
      </div>
      {action}
    </div>
  );
}

/**
 * A group of related settings.
 *
 * The icon and the description are not decoration: a settings screen that is
 * only labelled inputs makes the reader work out what each group is for.
 */
export function Section({
  icon: Icon,
  title,
  description,
  actions,
  children,
}: {
  icon?: LucideIcon;
  title: string;
  description?: string;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="flex flex-col gap-5 py-7">
      <div className="flex items-start gap-4">
        <div className="min-w-0 flex-1">
          <h2 className="flex items-center gap-2.5 text-base font-medium">
            {Icon !== undefined && <Icon className="text-muted-foreground size-[18px]" />}
            {title}
          </h2>
          {description !== undefined && (
            <p className="text-muted-foreground mt-1.5 text-sm leading-6">{description}</p>
          )}
        </div>
        {actions !== undefined && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
      </div>
      <div className="flex flex-col gap-5">{children}</div>
    </section>
  );
}

/**
 * One labelled control.
 *
 * Label above, helper below the label, control full width. The previous
 * label-left / control-right row could not carry a helper line at all, which
 * is why every non-obvious setting was left unexplained.
 */
export function Field({
  label,
  hint,
  htmlFor,
  children,
}: {
  label: string;
  hint?: string;
  htmlFor?: string;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-col gap-1">
        <label htmlFor={htmlFor} className="text-sm font-medium">
          {label}
        </label>
        {hint !== undefined && <p className="text-muted-foreground text-[13px]">{hint}</p>}
      </div>
      {children}
    </div>
  );
}

/** A row of facts inside a card — name on the left, value on the right. */
export function Stat({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="min-w-0 truncate text-right">{children}</span>
    </div>
  );
}

/**
 * A card in a collection grid.
 *
 * `footer` is the monospace line the reference puts at the bottom of every
 * card — the identifier you would use to find the thing outside the app.
 */
export function Card({
  title,
  status,
  children,
  footer,
  className,
}: {
  title: ReactNode;
  status?: ReactNode;
  children?: ReactNode;
  footer?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "border-border bg-card flex flex-col rounded-xl border transition-colors",
        "hover:border-muted-foreground/30",
        className,
      )}
    >
      <div className="flex flex-col gap-3 p-4">
        <div className="flex items-start justify-between gap-3">
          <h3 className="min-w-0 flex-1 font-medium">{title}</h3>
          {status !== undefined && <div className="shrink-0">{status}</div>}
        </div>
        {children !== undefined && (
          <div className="text-muted-foreground text-sm leading-6">{children}</div>
        )}
      </div>
      {footer !== undefined && (
        <div className="border-border/70 text-muted-foreground border-t px-4 py-2.5 font-mono text-xs">
          {footer}
        </div>
      )}
    </div>
  );
}

/** The reading column every page shares, so pages cannot drift apart. */
export function PageBody({ children, wide }: { children: ReactNode; wide?: boolean }) {
  return (
    <div className={cn("mx-auto w-full px-8 py-8", wide ? "max-w-5xl" : "max-w-3xl")}>
      {children}
    </div>
  );
}
